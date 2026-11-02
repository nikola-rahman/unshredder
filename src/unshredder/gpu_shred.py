"""Batched mechanical shredding + scan degradation on the GPU.

Page windows go in; the edge bands SeamNet trains on come out directly (full strips are
never materialised). Everything is randomised per page / per strip / per cut:

page   scale (scan dpi / font size), show-through from the back side, photocopy/fax
       degradation, toner streaks
cut    slanted and wobbly cut lines, kerf (lost material), frayed fibres (torn away /
       crushed), shadow along the paper edge
strip  vertical shift, vertical stretch and curl (paper not lying flat), broken ends,
       ink spread/thinning, blur, ink/paper levels, paper texture, sensor noise, JPEG

`strength` scales every probability and magnitude (0 = clean cuts only).
Works on any torch device; written for CUDA.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _u(gen, n, lo, hi, dev):
    return lo + (hi - lo) * torch.rand(n, generator=gen, device=dev)


def _p(gen, n, p, dev):
    return torch.rand(n, generator=gen, device=dev) < p


def _smooth_noise(gen, n: int, length: int, knots: int, dev) -> torch.Tensor:
    """(n, length) smooth random curves with unit-ish amplitude."""
    k = torch.randn(n, 1, knots, generator=gen, device=dev)
    return F.interpolate(k, size=length, mode="linear", align_corners=True)[:, 0]


# ---------------------------------------------------------------- page level


def augment_pages(pages: torch.Tensor, gen, strength: float) -> torch.Tensor:
    """pages (P, 1, H, W) in [0,1], 1 = paper."""
    P, _, H, W = pages.shape
    dev = pages.device
    s = strength
    # Show-through: faint mirrored text from "the back side" (another page of the batch).
    a = _u(gen, P, 0.02, 0.15, dev) * _p(gen, P, 0.3 * s, dev)
    ghost = torch.flip(pages[torch.randperm(P, generator=gen, device=dev)], dims=[3])
    pages = pages * (1 - a.view(P, 1, 1, 1) * (1 - ghost))
    # Photocopy / fax: resolution loss and contrast boost.
    low = _p(gen, P, 0.15 * s, dev)
    if low.any():
        f = int(torch.randint(2, 4, (1,), generator=gen, device=dev))
        lo_res = F.interpolate(F.avg_pool2d(pages, f, ceil_mode=True), size=(H, W), mode="bilinear")
        lo_res = torch.clamp((lo_res - 0.5) * 1.6 + 0.55, 0, 1)
        pages = torch.where(low.view(P, 1, 1, 1), lo_res, pages)
    # Toner streak: a faint vertical line along the paper feed direction.
    streak = _p(gen, P, 0.1 * s, dev)
    for p in torch.nonzero(streak).flatten().tolist():
        x = int(torch.randint(0, W - 4, (1,), generator=gen, device=dev))
        w = int(torch.randint(1, 4, (1,), generator=gen, device=dev))
        pages[p, :, :, x : x + w] *= float(_u(gen, 1, 0.75, 0.95, dev))
    return pages


# ---------------------------------------------------------------- cutting


def cut_bands(
    pages: torch.Tensor, n_strips: list[int], crop: int, band: int, max_shift: int, gen, strength: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cut each page into n strips and sample every strip's right and left edge band.

    Returns R, L (N, 1, crop, band) with the cut on the right (left bands mirrored),
    target (N,) = index of the strip's right neighbour or -1, and strip page ids (N,).
    """
    P, _, Hp, Wp = pages.shape
    dev = pages.device
    s = strength
    R_all, L_all, targets, owner = [], [], [], []
    yy = torch.arange(crop, device=dev, dtype=torch.float32)
    uu = torch.arange(band, device=dev, dtype=torch.float32)
    base_n = 0
    for p in range(P):
        n = n_strips[p]
        scale = float(_u(gen, 1, 1 - 0.25 * s, 1 + 0.3 * s, dev))  # dpi / font size
        Wc = Wp * scale  # page width in output pixels
        cuts = torch.linspace(0, Wc, n + 1, device=dev)
        cuts[1:-1] += (torch.rand(n - 1, generator=gen, device=dev) - 0.5) * 0.3 * Wc / n
        # Cut line x(y) per cut: slant + low-frequency wobble (outer page edges straight).
        slope = torch.randn(n + 1, generator=gen, device=dev) * 0.003 * s * crop
        wob = _smooth_noise(gen, n + 1, crop, 12, dev) * _u(gen, n + 1, 0, 1.5 * s, dev)[:, None]
        line = cuts[:, None] + slope[:, None] * (yy[None] / crop - 0.5) + wob
        line[0], line[-1] = 0.0, Wc
        kerf_l = _u(gen, n, 0, 2.5 * s, dev)
        kerf_r = _u(gen, n, 0, 2.5 * s, dev)
        kerf_l[0] = 0
        kerf_r[-1] = 0
        left_edge = line[:-1] + kerf_l[:, None]  # (n, crop) paper of strip i starts here
        right_edge = line[1:] - kerf_r[:, None]  # ... and ends here

        # Row mapping per strip: vertical shift, stretch and curl (paper not lying flat).
        dy = (torch.rand(n, generator=gen, device=dev) * 2 - 1) * max_shift
        stretch = 1 + (torch.rand(n, generator=gen, device=dev) * 2 - 1) * 0.015 * s
        curl = _smooth_noise(gen, n, crop, 6, dev) * _u(gen, n, 0, 3 * s, dev)[:, None]
        y_out = (yy[None] - crop / 2) * stretch[:, None] + crop / 2 + dy[:, None] + curl  # (n, crop)
        y_src = (y_out + max_shift + 4) / scale  # source page row (window has margin)

        # Band columns in output space, anchored on the rounded mean cut position.
        r_anchor = torch.round(right_edge.mean(1)) - band  # (n,)
        l_anchor = torch.round(left_edge.mean(1))
        xr = r_anchor[:, None] + uu[None]  # (n, band)
        xl = l_anchor[:, None] + uu[None]

        def sample(x_cols):  # (n, band) output-space columns -> (n, crop, band)
            gx = (x_cols[:, None, :] / scale) / (Wp - 1) * 2 - 1
            gy = y_src[:, :, None] / (Hp - 1) * 2 - 1
            grid = torch.stack([gx.expand(n, crop, band), gy.expand(n, crop, band)], -1)
            grid = grid.reshape(1, n * crop, band, 2)
            out = F.grid_sample(pages[p : p + 1], grid, mode="bilinear", padding_mode="border", align_corners=True)
            return out.reshape(n, crop, band)

        def finish(img, x_cols, edge_row_left, edge_row_right, cut_side):
            x = x_cols[:, None, :]
            outside = (x < edge_row_left[:, :, None]) | (x >= edge_row_right[:, :, None])
            img = torch.where(outside, torch.ones_like(img), img)
            # Fray the cut side: distance (px) from the cut, inside the paper.
            d = (edge_row_right[:, :, None] - x) if cut_side == "r" else (x - edge_row_left[:, :, None])
            depth = _u(gen, n, 1, 1 + 4 * s, dev).view(n, 1, 1)
            amount = _u(gen, n, 0.2, 0.7, dev).view(n, 1, 1) * min(s, 1.0)
            prob = amount * torch.exp(-d.clamp(min=0) / (depth / 2)) * (d >= 0) * (d < 3 * depth)
            u = torch.rand(img.shape, generator=gen, device=dev)
            img = torch.where(u < prob * 0.8, torch.ones_like(img), img)  # torn away
            crushed = u > 1 - prob * 0.15
            img = torch.where(crushed, torch.rand(img.shape, generator=gen, device=dev) * 0.5 + 0.2, img)
            # Shadow along the paper edge on the scanner glass.
            sh = _p(gen, n, 0.3 * s, dev).view(n, 1, 1) * _u(gen, n, 0.05, 0.3, dev).view(n, 1, 1)
            width = _u(gen, n, 1, 4, dev).view(n, 1, 1)
            ramp = torch.clamp(1 - d / width, 0, 1) * (d >= 0)
            return img * (1 - sh * ramp)

        Rb = finish(sample(xr), xr, left_edge, right_edge, "r")
        Lb = finish(sample(xl), xl, left_edge, right_edge, "l")

        # Broken strips: a missing top or bottom piece (both bands of the strip).
        broken = _p(gen, n, 0.15 * s, dev)
        if broken.any():
            cut_at = (torch.rand(n, generator=gen, device=dev) * 0.4 * crop).long()
            top = torch.rand(n, generator=gen, device=dev) < 0.5
            rows = torch.arange(crop, device=dev)[None]
            gone = broken[:, None] & torch.where(top[:, None], rows < cut_at[:, None], rows >= crop - cut_at[:, None])
            Rb = torch.where(gone[:, :, None], torch.ones_like(Rb), Rb)
            Lb = torch.where(gone[:, :, None], torch.ones_like(Lb), Lb)

        R_all.append(Rb)
        L_all.append(torch.flip(Lb, dims=[2]))  # mirror: cut on the right
        t = torch.arange(base_n + 1, base_n + n + 1, device=dev)
        t[-1] = -1
        targets.append(t)
        owner.append(torch.full((n,), p, device=dev))
        base_n += n
    R = torch.cat(R_all).unsqueeze(1)
    L = torch.cat(L_all).unsqueeze(1)
    return R, L, torch.cat(targets), torch.cat(owner)


# ---------------------------------------------------------------- strip level (scan)


def _dct_matrix(dev) -> torch.Tensor:
    k = torch.arange(8, device=dev, dtype=torch.float32)
    D = torch.cos((2 * k[None] + 1) * k[:, None] * math.pi / 16) * math.sqrt(2 / 8)
    D[0] /= math.sqrt(2)
    return D


_JPEG_Q = torch.tensor([
    [16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99],
], dtype=torch.float32)


def jpeg_like(x: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
    """Approximate JPEG luminance compression: 8x8 DCT, quantise, inverse. x (N,1,H,W)."""
    N, _, H, W = x.shape
    ph, pw = (-H) % 8, (-W) % 8
    xp = F.pad(x, (0, pw, 0, ph), mode="replicate") * 255 - 128
    Hb, Wb = xp.shape[2] // 8, xp.shape[3] // 8
    blocks = xp.reshape(N, Hb, 8, Wb, 8).permute(0, 1, 3, 2, 4)  # (N,Hb,Wb,8,8)
    D = _dct_matrix(x.device)
    coef = D @ blocks @ D.T
    q = quality.clamp(1, 100)
    scale = torch.where(q < 50, 5000 / q, 200 - 2 * q) / 100
    Q = (_JPEG_Q.to(x.device)[None] * scale.view(N, 1, 1)).clamp(min=1).view(N, 1, 1, 8, 8)
    coef = torch.round(coef / Q) * Q
    out = (D.T @ coef @ D).permute(0, 1, 3, 2, 4).reshape(N, 1, Hb * 8, Wb * 8)
    return ((out + 128) / 255)[:, :, :H, :W]


def _blur(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Per-sample separable gaussian blur, x (N,1,H,W), sigma (N,)."""
    N = x.shape[0]
    r = 3
    t = torch.arange(-r, r + 1, device=x.device, dtype=torch.float32)
    k = torch.exp(-(t[None] ** 2) / (2 * sigma.clamp(min=1e-3)[:, None] ** 2))
    k = k / k.sum(1, keepdim=True)
    xg = x.reshape(1, N, *x.shape[2:])
    xg = F.conv2d(F.pad(xg, (0, 0, r, r), mode="replicate"), k.view(N, 1, 2 * r + 1, 1), groups=N)
    xg = F.conv2d(F.pad(xg, (r, r, 0, 0), mode="replicate"), k.view(N, 1, 1, 2 * r + 1), groups=N)
    return xg.reshape_as(x)


def scan_augment(R: torch.Tensor, L: torch.Tensor, gen, strength: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-strip scanner/print effects, applied identically to a strip's two bands."""
    n = R.shape[0]
    dev = R.device
    s = strength
    x = torch.cat([R, L])  # (2n,1,H,W); strip i owns rows i and n+i

    def per_strip(v):
        return torch.cat([v, v])

    # Ink spread / thinning.
    mode = torch.randint(0, 3, (n,), generator=gen, device=dev) * _p(gen, n, 0.3 * s, dev)
    mode = per_strip(mode).view(-1, 1, 1, 1)
    dil = -F.max_pool2d(-F.pad(x, (0, 1, 0, 1), mode="replicate"), 2, stride=1)  # ink grows
    ero = F.max_pool2d(F.pad(x, (0, 1, 0, 1), mode="replicate"), 2, stride=1)  # ink thins
    x = torch.where(mode == 1, dil, torch.where(mode == 2, ero, x))
    # Blur.
    sig = per_strip(_u(gen, n, 0.3, 0.3 + 1.0 * s, dev) * _p(gen, n, 0.7 * s, dev))
    x = torch.where(sig.view(-1, 1, 1, 1) > 0.05, _blur(x, sig), x)
    # Ink / paper levels.
    lo = per_strip(_u(gen, n, 0, 0.25 * s, dev)).view(-1, 1, 1, 1)
    hi = per_strip(_u(gen, n, 1 - 0.2 * s, 1, dev)).view(-1, 1, 1, 1)
    x = lo + (hi - lo) * x
    # Paper texture / uneven illumination.
    H, W = x.shape[2:]
    tex = F.interpolate(torch.randn(2 * n, 1, max(2, H // 24), 3, generator=gen, device=dev), size=(H, W),
                        mode="bicubic", align_corners=False)
    amp = per_strip(_u(gen, n, 0.005, 0.03, dev) * _p(gen, n, 0.5 * s, dev)).view(-1, 1, 1, 1)
    x = x + tex * amp
    # Sensor noise.
    x = x + torch.randn(x.shape, generator=gen, device=dev) * per_strip(_u(gen, n, 0, 0.04 * s, dev)).view(-1, 1, 1, 1)
    x = x.clamp(0, 1)
    # JPEG.
    use = per_strip(_p(gen, n, 0.7 * s, dev))
    if use.any():
        q = per_strip(_u(gen, n, 40, 96, dev))
        x = torch.where(use.view(-1, 1, 1, 1), jpeg_like(x, q).clamp(0, 1), x)
    return x[:n], x[n:]


def normalize_bands(x: torch.Tensor) -> torch.Tensor:
    """Per-band paper level -> 1.0 (same idea as costs.normalize_strip)."""
    flat = x.flatten(1)
    k = max(1, int(flat.shape[1] * 0.95))
    bg = flat.kthvalue(k, dim=1).values.clamp(min=0.05).view(-1, 1, 1, 1)
    return (x / bg).clamp(0, 1)


def make_batch(pages: torch.Tensor, n_strips: list[int], crop: int, band: int, max_shift: int, gen,
               strength: float):
    """Full GPU pipeline: page windows -> (R, L, target) ready for the contrastive loss."""
    pages = augment_pages(pages, gen, strength)
    R, L, target, _ = cut_bands(pages, n_strips, crop, band, max_shift, gen, strength)
    R, L = scan_augment(R, L, gen, strength)
    return normalize_bands(R), normalize_bands(L), target


# ---------------------------------------------------------------- full strips (benchmarks)


@torch.no_grad()
def cut_strips_full(page: torch.Tensor, n: int, gen, strength: float, max_shift: int = 12,
                    stretch: float = 0.015) -> list[torch.Tensor]:
    """Shred one whole page (1, 1, H, W) into n full-width degraded strips, left to right.

    Same physics as training (cut geometry, kerf, fraying, shift/stretch/curl, broken
    ends, scan effects), but every strip is kept whole, at native scale, for building
    realistic piles to reconstruct. Returns (H, w_i) tensors in [0, 1]."""
    dev = page.device
    s = strength
    page = augment_pages(page, gen, s)
    m = max_shift + 4
    padded = F.pad(page, (0, 0, m, m), value=1.0)
    _, _, Hp, Wp = padded.shape
    H = page.shape[2]
    yy = torch.arange(H, device=dev, dtype=torch.float32)

    cuts = torch.linspace(0, Wp, n + 1, device=dev)
    cuts[1:-1] += (torch.rand(n - 1, generator=gen, device=dev) - 0.5) * 0.3 * Wp / n
    slope = torch.randn(n + 1, generator=gen, device=dev) * 0.003 * s * H
    wob = _smooth_noise(gen, n + 1, H, 40, dev) * _u(gen, n + 1, 0, 1.5 * s, dev)[:, None]
    line = cuts[:, None] + slope[:, None] * (yy[None] / H - 0.5) + wob
    line[0], line[-1] = 0.0, float(Wp)
    kerf_l, kerf_r = _u(gen, n, 0, 2.5 * s, dev), _u(gen, n, 0, 2.5 * s, dev)
    kerf_l[0] = kerf_r[-1] = 0
    left_edge = line[:-1] + kerf_l[:, None]
    right_edge = line[1:] - kerf_r[:, None]

    dy = (torch.rand(n, generator=gen, device=dev) * 2 - 1) * max_shift
    stretch = 1 + (torch.rand(n, generator=gen, device=dev) * 2 - 1) * stretch * s
    curl = _smooth_noise(gen, n, H, 12, dev) * _u(gen, n, 0, 3 * s, dev)[:, None]
    y_src = (yy[None] - H / 2) * stretch[:, None] + H / 2 + dy[:, None] + curl + m  # (n, H)

    x0 = torch.floor(left_edge.min(1).values).clamp(min=0)
    x1 = torch.ceil(right_edge.max(1).values).clamp(max=Wp)
    width = int((x1 - x0).max())
    uu = torch.arange(width, device=dev, dtype=torch.float32)
    xs = x0[:, None] + uu[None]  # (n, width)
    gx = (xs[:, None, :] / (Wp - 1) * 2 - 1).expand(n, H, width)
    gy = (y_src[:, :, None] / (Hp - 1) * 2 - 1).expand(n, H, width)
    grid = torch.stack([gx, gy], -1).reshape(1, n * H, width, 2)
    img = F.grid_sample(padded, grid, mode="bilinear", padding_mode="border", align_corners=True).reshape(n, H, width)

    x = xs[:, None, :]
    img = torch.where((x < left_edge[:, :, None]) | (x >= right_edge[:, :, None]), torch.ones_like(img), img)
    for d in (right_edge[:, :, None] - x, x - left_edge[:, :, None]):  # fray both cuts
        depth = _u(gen, n, 1, 1 + 4 * s, dev).view(n, 1, 1)
        amount = _u(gen, n, 0.2, 0.7, dev).view(n, 1, 1) * min(s, 1.0)
        prob = amount * torch.exp(-d.clamp(min=0) / (depth / 2)) * (d >= 0) * (d < 3 * depth)
        u = torch.rand(img.shape, generator=gen, device=dev)
        img = torch.where(u < prob * 0.8, torch.ones_like(img), img)
        img = torch.where(u > 1 - prob * 0.15, torch.rand(img.shape, generator=gen, device=dev) * 0.5 + 0.2, img)

    broken = _p(gen, n, 0.15 * s, dev)
    if broken.any():
        cut_at = (torch.rand(n, generator=gen, device=dev) * 0.4 * H).long()
        top = torch.rand(n, generator=gen, device=dev) < 0.5
        rows = torch.arange(H, device=dev)[None]
        gone = broken[:, None] & torch.where(top[:, None], rows < cut_at[:, None], rows >= H - cut_at[:, None])
        img = torch.where(gone[:, :, None], torch.ones_like(img), img)

    img, _ = scan_augment(img[:, None], img[:, None], gen, s)
    return [img[i, 0, :, : int(x1[i] - x0[i])] for i in range(n)]
