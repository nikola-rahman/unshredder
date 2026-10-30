import io
import json
import time

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from unshredder import server


def _png(w=60, h=80):
    buf = io.BytesIO()
    Image.fromarray((np.random.rand(h, w) * 255).astype(np.uint8)).save(buf, "PNG")
    return buf.getvalue()


def _wait(client, job, states=("scanned", "shredded", "done", "failed")):
    for _ in range(600):
        s = client.get(f"/jobs/{job}").json()
        if s["state"] in states:
            return s
        time.sleep(0.05)
    raise AssertionError(s)


def _page(w=400, h=560, seed=0):
    rng = np.random.default_rng(seed)
    a = np.full((h, w), 255, np.uint8)
    for y in range(40, h - 40, 24):  # "text lines" of random dark blocks
        for x in range(30, w - 30, 9):
            if rng.random() < 0.6:
                a[y:y + 12, x:x + 6] = 20
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, "PNG")
    return buf.getvalue()


def test_scan_shred_unshred_review(tmp_path, monkeypatch):
    def fake_scores(strips, model_path, max_shift):  # no model: random scores
        n = len(strips)
        return np.random.default_rng(0).normal(size=(n, n)).astype(np.float32), np.full((n, 1), 1.0)

    monkeypatch.setattr(server.seamnet, "scores", fake_scores)
    client = TestClient(server.create_app("m.pt", str(tmp_path)))
    files = [("files", ("a.png", _page(seed=1), "image/png")), ("files", ("b.png", _page(seed=2), "image/png")),
             ("files", ("notes.txt", b"x", "text/plain"))]
    job = client.post("/jobs", files=files, data={"mode": "scan"}).json()["job"]
    s = _wait(client, job)
    assert s["state"] == "scanned", s
    assert [p["name"] for p in s["scan"]["pages"]] == ["a", "b"]
    assert client.get(f"/jobs/{job}/scans/0.png").status_code == 200
    assert client.post(f"/jobs/{job}/unshred").status_code == 409  # not shredded yet

    assert client.post(f"/jobs/{job}/shred", data={"strips_per_page": 8, "damage": 0.3}).status_code == 200
    s = _wait(client, job, ("shredded", "failed"))
    assert s["state"] == "shredded", s
    assert s["shred"]["strips"] == 16
    assert client.get(f"/jobs/{job}/pile.png").status_code == 200

    assert client.post(f"/jobs/{job}/unshred").status_code == 200
    s = _wait(client, job, ("done", "failed"))
    assert s["state"] == "done", s
    assert sum(p["strips"] for p in s["result"]["pieces"]) == 16 - s["result"]["blank_strips"]
    st = client.post(f"/jobs/{job}/review/start").json()
    assert "test" in st and st["question"]["options"]
    q = st["question"]
    assert client.post(f"/jobs/{job}/review/answer", data={"anchor": q["anchor"], "choice": q["options"][0]["id"]}).status_code == 200

    # re-shredding with other settings resets the downstream results
    client.post(f"/jobs/{job}/shred", data={"strips_per_page": 4})
    s = _wait(client, job, ("shredded", "failed"))
    assert s["shred"]["strips"] == 8 and "result" not in s
    assert client.post(f"/jobs/{job}/review/start").status_code == 409


def test_strips_job_and_bad_input(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "unshred_job", lambda store, d, t: store.set(d, state="done", step="done"))
    client = TestClient(server.create_app("m.pt", str(tmp_path)))
    files = [("files", (f"{k}.png", _png(), "image/png")) for k in range(3)]
    job = client.post("/jobs", files=files, data={"mode": "strips"}).json()["job"]
    s = _wait(client, job, ("done", "failed"))
    assert s["shred"]["strips"] == 3
    assert client.get("/jobs/nope").status_code == 404
    assert client.post("/jobs", files=files, data={"mode": "bad"}).status_code == 400
    assert client.post("/jobs", files=[("files", ("x.txt", b"x", "text/plain"))]).status_code == 400


def test_render_chain_marks_uncertain_seams():
    imgs = [np.ones((50, 10), np.float32), np.ones((50, 12), np.float32)]
    img = server.render_chain([(0, 0), (1, 0)], imgs, [0.2, None])
    px = np.asarray(img)
    assert (px[:, 9] == [220, 30, 30]).all()  # red mark at the uncertain seam


def test_review_start_keeps_answers(tmp_path, monkeypatch):
    def fake_scores(strips, model_path, max_shift):
        n = len(strips)
        return np.random.default_rng(1).normal(size=(n, n)).astype(np.float32), np.full((n, 1), 1.0)

    monkeypatch.setattr(server.seamnet, "scores", fake_scores)
    client = TestClient(server.create_app("m.pt", str(tmp_path)))
    job = client.post("/jobs", files=[("files", ("a.png", _page(seed=3), "image/png"))], data={"mode": "scan"}).json()["job"]
    _wait(client, job)
    client.post(f"/jobs/{job}/shred", data={"strips_per_page": 8})
    _wait(client, job, ("shredded", "failed"))
    client.post(f"/jobs/{job}/unshred")
    _wait(client, job, ("done", "failed"))
    q = client.post(f"/jobs/{job}/review/start").json()["question"]
    client.post(f"/jobs/{job}/review/answer", data={"anchor": q["anchor"], "choice": q["options"][0]["id"]})
    assert client.post(f"/jobs/{job}/review/start").json()["progress"]["your_decisions"] == 1
    assert client.post(f"/jobs/{job}/review/start", data={"restart": "true"}).json()["progress"]["your_decisions"] == 0


def test_cut_wrong_join_persists_and_undoes(tmp_path, monkeypatch):
    def fake_scores(strips, model_path, max_shift):  # very confident: everything auto-joins
        n = len(strips)
        S = np.full((n, n), -20.0, np.float32)
        for a in range(n - 1):
            S[a, a + 1] = 20.0
        return S, np.full((n, 1), -20.0)

    monkeypatch.setattr(server.seamnet, "scores", fake_scores)
    client = TestClient(server.create_app("m.pt", str(tmp_path)))
    job = client.post("/jobs", files=[("files", ("a.png", _page(seed=4), "image/png"))], data={"mode": "scan"}).json()["job"]
    _wait(client, job)
    client.post(f"/jobs/{job}/shred", data={"strips_per_page": 8})
    _wait(client, job, ("shredded", "failed"))
    client.post(f"/jobs/{job}/unshred")
    _wait(client, job, ("done", "failed"))
    client.post(f"/jobs/{job}/review/start")
    fin = client.post(f"/jobs/{job}/review/finish").json()
    seam = next(sm for p in fin["pieces"] for sm in p["seams"])
    assert 0 < seam["x"] < 1
    r = client.post(f"/jobs/{job}/review/cut", data={"left": seam["left"], "right": seam["right"]})
    assert r.status_code == 200 and r.json()["progress"]["your_decisions"] == 1
    assert client.post(f"/jobs/{job}/review/cut", data={"left": seam["left"], "right": seam["right"]}).status_code == 409
    # survives a server restart (fresh app on the same jobs folder), and cannot come back
    client2 = TestClient(server.create_app("m.pt", str(tmp_path)))
    st = client2.post(f"/jobs/{job}/review/start").json()
    assert st["progress"]["your_decisions"] == 1
    pairs = {(sm["left"], sm["right"]) for p in client2.post(f"/jobs/{job}/review/finish").json()["pieces"] for sm in p["seams"]}
    assert (seam["left"], seam["right"]) not in pairs
    assert client2.post(f"/jobs/{job}/review/undo").json()["progress"]["your_decisions"] == 0


def test_ask_later_and_remove_any_answer(tmp_path, monkeypatch):
    def fake_scores(strips, model_path, max_shift):
        n = len(strips)
        return np.random.default_rng(2).normal(size=(n, n)).astype(np.float32), np.full((n, 1), 1.0)

    monkeypatch.setattr(server.seamnet, "scores", fake_scores)
    client = TestClient(server.create_app("m.pt", str(tmp_path)))
    job = client.post("/jobs", files=[("files", ("a.png", _page(seed=5), "image/png"))], data={"mode": "scan"}).json()["job"]
    _wait(client, job)
    client.post(f"/jobs/{job}/shred", data={"strips_per_page": 10})
    _wait(client, job, ("shredded", "failed"))
    client.post(f"/jobs/{job}/unshred")
    _wait(client, job, ("done", "failed"))
    st = client.post(f"/jobs/{job}/review/start").json()
    first = st["question"]["anchor"]
    st = client.post(f"/jobs/{job}/review/later", data={"anchor": first}).json()
    assert st["later"] == 1 and st["question"]["anchor"] != first and st["progress"]["your_decisions"] == 0
    # answer two questions, then take back the first of them only
    for _ in range(2):
        q = st["question"]
        st = client.post(f"/jobs/{job}/review/answer", data={"anchor": q["anchor"], "choice": q["options"][0]["id"]}).json()
    assert [x["i"] for x in st["answers"]] == [0, 1]
    kept = st["answers"][1]
    st = client.post(f"/jobs/{job}/review/remove", data={"i": 0}).json()
    assert len(st["answers"]) == 1 and (st["answers"][0]["a"], st["answers"][0]["b"]) == (kept["a"], kept["b"])
    # survives a restart; the later-question comes back once nothing else is open
    st = TestClient(server.create_app("m.pt", str(tmp_path))).post(f"/jobs/{job}/review/start").json()
    assert st["later"] == 1 and len(st["answers"]) == 1


def test_pieces_and_focus_left_and_right(tmp_path, monkeypatch):
    def fake_scores(strips, model_path, max_shift):
        n = len(strips)
        return np.random.default_rng(3).normal(size=(n, n)).astype(np.float32), np.full((n, 1), 0.5)

    monkeypatch.setattr(server.seamnet, "scores", fake_scores)
    client = TestClient(server.create_app("m.pt", str(tmp_path)))
    job = client.post("/jobs", files=[("files", ("a.png", _page(seed=6), "image/png"))], data={"mode": "scan"}).json()["job"]
    _wait(client, job)
    client.post(f"/jobs/{job}/shred", data={"strips_per_page": 10})
    _wait(client, job, ("shredded", "failed"))
    client.post(f"/jobs/{job}/unshred")
    _wait(client, job, ("done", "failed"))
    client.post(f"/jobs/{job}/review/start")
    pcs = client.get(f"/jobs/{job}/review/pieces").json()
    assert pcs["pieces"] and all(0 < p["width_share"] <= 1 and p["joins"] == p["strips"] - 1 for p in pcs["pieces"])
    assert all("joins_correct" in p for p in pcs["pieces"])
    assert client.get(f"/jobs/{job}/review/piece_thumb.png", params={"first": pcs["pieces"][0]["first"]}).status_code == 200
    f = pcs["pieces"][0]["first"]
    st = client.get(f"/jobs/{job}/review/state", params={"focus": f}).json()
    assert st["focus"]["strips"] >= 1 and st["question"]["side"] == "right"
    # close the right end, then the left question appears; mark the page start -> piece done
    st = client.post(f"/jobs/{job}/review/answer", data={"anchor": st["question"]["anchor"], "choice": "end", "focus": f}).json()
    q = st["question"]
    assert q["side"] == "left" and q["anchor"] == f and any(o["id"] == "start" for o in q["options"])
    cand = next(o["id"] for o in q["options"] if o["id"] != "start")
    st = client.post(f"/jobs/{job}/review/answer", data={"anchor": f, "choice": cand, "side": "left", "focus": f}).json()
    assert st["focus"]["strips"] >= 2  # the piece grew to the left
    while st.get("question"):
        q = st["question"]
        edge = "start" if q["side"] == "left" else "end"
        st = client.post(f"/jobs/{job}/review/answer", data={"anchor": q["anchor"], "choice": edge, "side": q["side"], "focus": f}).json()
    assert st["focus"]["done"]
    assert any(a["kind"] == "start" for a in st["answers"])
    # persisted across restarts
    st2 = TestClient(server.create_app("m.pt", str(tmp_path))).get(f"/jobs/{job}/review/state", params={"focus": f}).json()
    assert st2["focus"]["done"]


def test_finished_pieces_listed_separately(tmp_path, monkeypatch):
    def fake_scores(strips, model_path, max_shift):
        n = len(strips)
        return np.random.default_rng(4).normal(size=(n, n)).astype(np.float32), np.full((n, 1), 0.5)

    monkeypatch.setattr(server.seamnet, "scores", fake_scores)
    client = TestClient(server.create_app("m.pt", str(tmp_path)))
    job = client.post("/jobs", files=[("files", ("a.png", _page(seed=7), "image/png"))], data={"mode": "scan"}).json()["job"]
    _wait(client, job)
    client.post(f"/jobs/{job}/shred", data={"strips_per_page": 10})
    _wait(client, job, ("shredded", "failed"))
    client.post(f"/jobs/{job}/unshred")
    _wait(client, job, ("done", "failed"))
    client.post(f"/jobs/{job}/review/start")
    f = client.get(f"/jobs/{job}/review/pieces").json()["pieces"][0]["first"]
    st = client.get(f"/jobs/{job}/review/state", params={"focus": f}).json()
    while st.get("question"):  # close both ends of that piece
        q = st["question"]
        edge = "start" if q["side"] == "left" else "end"
        st = client.post(f"/jobs/{job}/review/answer", data={"anchor": q["anchor"], "choice": edge, "side": q["side"], "focus": f}).json()
    r = client.get(f"/jobs/{job}/review/pieces").json()
    assert r["finished_total"] == 1 and r["todo_total"] == r["total"] - 1
    assert [p["finished"] for p in r["pieces"]][-1] is True and not any(p["finished"] for p in r["pieces"][:-1])
