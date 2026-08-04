from app.web_paths import (
    api_settings_path,
    append_query,
    document_workbench_path,
    safe_next,
    work_archive_destination,
    workbench_path,
)


def test_safe_next_accepts_only_local_paths():
    assert safe_next("/dashboard?tab=models") == "/dashboard?tab=models"
    assert safe_next("https://example.com", "/dashboard") == "/dashboard"
    assert safe_next("//example.com", "/dashboard") == "/dashboard"
    assert safe_next("/safe\\redirect", "/dashboard") == "/dashboard"
    assert safe_next("/safe\nredirect", "/dashboard") == "/dashboard"


def test_workbench_paths_encode_identity_and_state():
    assert workbench_path(
        "project/with spaces",
        chapter_id="chapter 1",
        conversation_id="conversation 1",
    ) == (
        "/novels/project%2Fwith%20spaces/workbench"
        "?chapter_id=chapter+1&conversation_id=conversation+1"
    )
    assert document_workbench_path(
        "document/one",
        chapter_id="chapter 1",
        view="archive",
    ) == ("/documents/document%2Fone?chapter_id=chapter+1&view=archive")


def test_settings_and_archive_destinations_share_one_url_builder():
    assert api_settings_path(
        return_to="/novels/p/workbench?chapter_id=c",
        tab="routing",
    ) == (
        "/settings/api?tab=routing&return_to=%2Fnovels%2Fp%2Fworkbench%3Fchapter_id%3Dc"
    )
    assert append_query("/dashboard?view=archive", saved="true") == (
        "/dashboard?view=archive&saved=true"
    )
    assert work_archive_destination(
        {
            "current_version": {
                "project_id": "project-1",
            }
        },
        saved=True,
    ) == ("/novels/project-1/workbench?view=archive&archive_tab=creative&saved=true")
