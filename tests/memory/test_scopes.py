from hivi.memory.scopes import bank_id_for_project, normalize_project_id


def test_normalize_project_id():
    assert normalize_project_id("1") == "001"
    assert normalize_project_id("001") == "001"
    assert normalize_project_id("project_001") == "001"
    assert normalize_project_id("project-12") == "012"


def test_bank_id_for_project():
    assert bank_id_for_project("001") == "hivi_project_001"
    assert bank_id_for_project("1") == "hivi_project_001"
