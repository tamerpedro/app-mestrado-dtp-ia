import hashlib

from src.eval_mode import is_modo_avaliacao, library_hash


def test_modo_avaliacao_defaults_to_true(monkeypatch):
    monkeypatch.delenv("MODO_AVALIACAO", raising=False)

    assert is_modo_avaliacao() is True


def test_modo_avaliacao_can_be_disabled_by_env(monkeypatch):
    monkeypatch.setenv("MODO_AVALIACAO", "false")

    assert is_modo_avaliacao() is False


def test_modo_avaliacao_accepts_truthy_variants(monkeypatch):
    for value in ["1", "true", "True", "yes", "on"]:
        monkeypatch.setenv("MODO_AVALIACAO", value)
        assert is_modo_avaliacao() is True


def test_library_hash_is_stable_sha256_hex(tmp_path):
    path = tmp_path / "riscos.csv"
    content = b"id,titulo\nR001,Teste\n"
    path.write_bytes(content)
    expected = hashlib.sha256(content).hexdigest()

    assert library_hash(path) == expected
    assert library_hash(path) == library_hash(path)


def test_library_hash_changes_with_content(tmp_path):
    path_a = tmp_path / "riscos_a.csv"
    path_b = tmp_path / "riscos_b.csv"
    path_a.write_text("id,titulo\nR001,Um\n", encoding="utf-8")
    path_b.write_text("id,titulo\nR001,Dois\n", encoding="utf-8")

    assert library_hash(path_a) != library_hash(path_b)
