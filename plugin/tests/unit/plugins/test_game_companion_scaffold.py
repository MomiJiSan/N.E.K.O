from pathlib import Path
from typing import Any

from PIL import Image

from plugin.plugins.game_companion import GameCompanionPlugin
from plugin.plugins.game_companion.core.frame_analyzer import analyze_frame
from plugin.plugins.game_companion.core.profile_registry import (
    ProfileMetadata,
    ProfileRegistry,
)
from plugin.plugins.game_companion.core.realtime import RealtimeInsightSession
from plugin.plugins.game_companion.profiles import builtin_profiles
from plugin.plugins.game_companion.profiles.tft.insights import (
    FORBIDDEN_DIRECTIVE_TERMS,
    generate_insights,
)
from plugin.plugins.game_companion.profiles.tft.recognition import recognize_shop_units
from plugin.plugins.game_companion.profiles.tft.screen_regions import (
    LAYOUT_AUGMENT_SELECT,
    LAYOUT_NORMAL_SHOP,
    LAYOUT_SPECIAL,
    LAYOUT_STATES,
    SHOP_SLOT_KEYS,
    UnsupportedAspectRatioError,
    grouped_screen_region_metadata,
    grouped_screen_region_bboxes,
    save_debug_crops,
)
from plugin.plugins.game_companion.profiles.tft.state_parser import parse_tft_state
from plugin.plugins.game_companion.safety.models import (
    Capability,
    CapabilityGate,
    GameType,
    RuntimeMode,
)


def _entry_contract_plugin() -> GameCompanionPlugin:
    plugin = object.__new__(GameCompanionPlugin)
    plugin._profiles = ProfileRegistry()
    for profile in builtin_profiles():
        plugin._profiles.register(profile)
    plugin._active_profile_id = "generic"
    plugin._realtime = RealtimeInsightSession()
    return plugin


def _payload(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value


def _profile_ids(payload: dict[str, Any]) -> set[str]:
    return {str(profile["id"]) for profile in payload["profiles"]}


def test_builtin_profiles_are_exact_phase0_set() -> None:
    profiles = {profile.profile_id: profile for profile in builtin_profiles()}

    assert set(profiles) == {"generic", "galgame", "tft"}
    assert "lol" not in profiles
    assert profiles["tft"].game_type is GameType.TYPE_D
    assert profiles["tft"].default_runtime_mode is RuntimeMode.ONLINE


def test_profile_registry_normalizes_and_orders_ids() -> None:
    registry = ProfileRegistry()
    registry.register(ProfileMetadata(profile_id="TFT", display_name="TFT", game_type=GameType.TYPE_D))
    registry.register(ProfileMetadata(profile_id="generic", display_name="Generic", game_type=GameType.TYPE_D))

    assert registry.has("tft")
    assert registry.has(" TFT ")
    assert [profile.profile_id for profile in registry.list()] == ["generic", "tft"]


def test_profile_registry_rejects_duplicates() -> None:
    registry = ProfileRegistry()
    registry.register(ProfileMetadata(profile_id="tft", display_name="TFT", game_type=GameType.TYPE_D))

    try:
        registry.register(ProfileMetadata(profile_id="TFT", display_name="TFT again", game_type=GameType.TYPE_D))
    except ValueError as exc:
        assert "duplicate profile_id" in str(exc)
    else:
        raise AssertionError("duplicate profile id was accepted")


def test_status_and_list_profiles_entry_contract() -> None:
    plugin = _entry_contract_plugin()

    status = _payload(plugin.status())
    listed = _payload(plugin.list_profiles())

    assert status["status"] == "ready"
    assert status["active_profile"] == "generic"
    assert _profile_ids(status) == {"generic", "galgame", "tft"}
    assert _profile_ids(listed) == {"generic", "galgame", "tft"}

    profiles = {profile["id"]: profile for profile in status["profiles"]}
    assert profiles["tft"]["display_name"] == "Teamfight Tactics"
    assert profiles["tft"]["game_type"] == GameType.TYPE_D.value
    assert profiles["tft"]["default_runtime_mode"] == RuntimeMode.ONLINE.value
    assert "vision_classify" in profiles["tft"]["capabilities"]


def test_select_tft_updates_status_entry_contract() -> None:
    plugin = _entry_contract_plugin()

    selected = _payload(plugin.select_profile(" TFT "))
    status = _payload(plugin.status())

    assert selected == {"selected": True, "active_profile": "tft"}
    assert status["active_profile"] == "tft"


def test_select_unknown_profile_keeps_previous_active_profile() -> None:
    plugin = _entry_contract_plugin()
    _payload(plugin.select_profile("tft"))

    rejected = _payload(plugin.select_profile("lol"))

    assert rejected["selected"] is False
    assert rejected["active_profile"] == "tft"
    assert rejected["error"] == "unknown_profile"
    assert set(rejected["available_profiles"]) == {"generic", "galgame", "tft"}
    assert _payload(plugin.status())["active_profile"] == "tft"


def test_capability_gate_defaults_to_read_only_observation() -> None:
    gate = CapabilityGate()

    assert gate.allows(Capability.SCREEN_OBSERVE)
    assert gate.allows(Capability.OCR)
    assert gate.allows(Capability.VISION_CLASSIFY)
    assert gate.allows(Capability.NEKO_CONTEXT)
    assert gate.denies(Capability.INPUT_CONTROL)
    assert gate.denies(Capability.AUTO_CLICK)
    assert gate.denies(Capability.MEMORY_READ)
    assert gate.denies(Capability.PACKET_READ)
    assert gate.denies(Capability.AUTOMATED_GAMEPLAY)


def test_profile_metadata_rejects_denied_capabilities() -> None:
    try:
        ProfileMetadata(
            profile_id="bad",
            display_name="Bad",
            game_type=GameType.TYPE_D,
            capabilities=(Capability.INPUT_CONTROL,),
        )
    except ValueError as exc:
        assert "denied by gate" in str(exc)
    else:
        raise AssertionError("denied capability was accepted")


def test_tft_regions_for_1920x1080() -> None:
    regions = grouped_screen_region_bboxes(1920, 1080)

    assert set(regions) == {
        "shop_slots",
        "shop",
        "bench",
        "board",
        "buy_xp_button",
        "equipment",
        "gold",
        "items_area",
        "level",
        "level_exp",
        "notifications",
        "players_panel",
        "stage",
        "round",
        "augments",
        "refresh_button",
        "shop_odds",
        "traits_panel",
    }
    assert len(regions["shop_slots"]) == 5
    assert regions["shop"] == (470, 878, 1422, 1046)
    assert regions["gold"] == (920, 842, 1008, 892)


def test_tft_layout_metadata_covers_all_layout_states() -> None:
    payload = grouped_screen_region_metadata(1920, 1080)

    assert set(payload["layout_profiles"]) == set(LAYOUT_STATES)
    assert set(payload["groups"]) >= {*LAYOUT_STATES, "shop_slots"}
    for layout in LAYOUT_STATES:
        profile = payload["layout_profiles"][layout]
        group = payload["groups"][layout]
        expected_keys = []
        for key in profile["primary_regions"]:
            if key == "shop_slots":
                expected_keys.extend(SHOP_SLOT_KEYS)
            elif key in payload["regions"]:
                expected_keys.append(key)
        for key, region in payload["regions"].items():
            if layout in region["active_layouts"]:
                expected_keys.append(key)
        expected_keys = list(dict.fromkeys(expected_keys))
        assert profile["key"] == layout
        assert group
        assert [region["key"] for region in group] == expected_keys
        assert all(region["bbox"] for region in group)

    assert payload["layout_profiles"][LAYOUT_NORMAL_SHOP]["deep_recognition"] is True
    assert payload["layout_profiles"][LAYOUT_SPECIAL]["deep_recognition"] is False


def test_tft_region_metadata_includes_layout_purpose_and_recognizers() -> None:
    payload = grouped_screen_region_metadata(1920, 1080)

    gold = payload["regions"]["gold"]
    assert gold["display_name"] == "Gold"
    assert gold["layout"] == LAYOUT_NORMAL_SHOP
    assert gold["priority"] == 2
    assert gold["purpose"] == "economy"
    assert gold["recognizers"] == ["ocr"]
    assert gold["bbox"] == [920, 842, 1008, 892]
    assert gold["ratio_bbox"] == [920 / 1920, 842 / 1080, 1008 / 1920, 892 / 1080]

    augments = payload["regions"]["augments"]
    assert augments["layout"] == LAYOUT_AUGMENT_SELECT
    assert augments["priority"] == 1
    assert augments["purpose"] == "augment_text"
    assert augments["recognizers"] == ["ocr"]

    assert [region["key"] for region in payload["groups"]["shop_slots"]] == list(SHOP_SLOT_KEYS)


def test_tft_regions_scale_to_other_16_9_resolutions() -> None:
    width, height = 1280, 720
    regions = grouped_screen_region_bboxes(width, height)

    shop_left, shop_top, shop_right, shop_bottom = regions["shop"]
    assert 0.24 * width <= shop_left <= 0.25 * width
    assert 0.81 * height <= shop_top <= 0.82 * height
    assert 0.74 * width <= shop_right <= 0.75 * width
    assert 0.96 * height <= shop_bottom <= 0.97 * height

    board_left, board_top, board_right, board_bottom = regions["board"]
    assert 0.18 * width <= board_left <= 0.19 * width
    assert 0.17 * height <= board_top <= 0.18 * height
    assert 0.81 * width <= board_right <= 0.82 * width
    assert 0.68 * height <= board_bottom <= 0.69 * height

    assert regions["gold"] == (613, 561, 672, 595)
    assert regions["shop_slots"]["shop_slot_1"] == (316, 585, 433, 697)


def test_tft_regions_reject_non_16_9() -> None:
    try:
        grouped_screen_region_bboxes(1280, 1024)
    except UnsupportedAspectRatioError as exc:
        assert "16:9" in str(exc)
    else:
        raise AssertionError("non-16:9 screenshot was accepted")


def test_analyze_frame_reads_tft_image(tmp_path: Path) -> None:
    screenshot = tmp_path / "tft.png"
    Image.new("RGB", (1920, 1080), color=(10, 20, 30)).save(screenshot)

    payload = analyze_frame("tft", screenshot)

    assert payload["success"] is True
    assert payload["ok"] is True
    assert payload["profile"] == "tft"
    assert payload["source"]["type"] == "image_path"
    assert payload["source"]["width"] == 1920
    assert payload["source"]["height"] == 1080
    assert payload["state"]["stage"] is None
    assert payload["state"]["level"] is None
    assert payload["state"]["gold"] is None
    assert payload["state"]["board_units"] == []
    assert payload["state"]["bench_units"] == []
    assert len(payload["state"]["shop_units"]) == 5
    assert payload["state"]["items"] == []
    assert payload["state"]["traits"] == []
    assert len(payload["regions"]["shop_slots"]) == 5
    assert isinstance(payload["insights"], list)
    assert payload["diagnostics"]["ocr"]["status"] in {"ready", "unavailable", "failed"}
    assert payload["diagnostics"]["recognition"]["status"] in {
        "ready",
        "no_templates",
        "imagehash_unavailable",
    }


def test_analyze_frame_debug_crops_include_expected_names_and_metadata(tmp_path: Path) -> None:
    screenshot = tmp_path / "tft.png"
    Image.new("RGB", (1920, 1080), color=(10, 20, 30)).save(screenshot)

    payload = analyze_frame("tft", screenshot, debug_crops_dir=tmp_path / "debug_crops")

    debug_crops = payload["diagnostics"]["debug_crops"]
    assert payload["success"] is True
    assert debug_crops is not None
    assert set(debug_crops["layout_profiles"]) == set(LAYOUT_STATES)
    assert Path(debug_crops["output_dir"]).is_dir()

    expected_names = {
        "gold": "normal_shop__p02__gold.png",
        "shop_slot_1": "normal_shop__p04__shop_slot_1.png",
        "board": "normal_shop__p08__board.png",
        "augments": "augment_select__p01__augments.png",
    }
    expected_regions = grouped_screen_region_bboxes(1920, 1080)
    for key, filename in expected_names.items():
        crop_path = Path(debug_crops["crops"][key])
        metadata = debug_crops["metadata"][key]
        expected_bbox = expected_regions["shop_slots"][key] if key.startswith("shop_slot_") else expected_regions[key]
        assert crop_path.name == filename
        assert crop_path.exists()
        assert metadata["crop_path"] == str(crop_path.resolve())
        assert metadata["bbox"] == list(expected_bbox)
        with Image.open(crop_path) as crop:
            left, top, right, bottom = metadata["bbox"]
            assert crop.size == (right - left, bottom - top)

    assert debug_crops["metadata"]["augments"]["layout"] == LAYOUT_AUGMENT_SELECT
    assert debug_crops["metadata"]["gold"]["purpose"] == "economy"


def test_save_debug_crops_flattens_grouped_regions_with_metadata(tmp_path: Path) -> None:
    screenshot = tmp_path / "tft.png"
    Image.new("RGB", (1920, 1080), color=(10, 20, 30)).save(screenshot)

    payload = save_debug_crops(screenshot, tmp_path / "crops")

    assert set(payload["layout_profiles"]) == set(LAYOUT_STATES)
    assert Path(payload["crops"]["shop_slot_5"]).name == "normal_shop__p04__shop_slot_5.png"
    assert payload["metadata"]["shop_slot_5"]["bbox"] == [1242, 878, 1418, 1046]
    assert payload["metadata"]["shop_slot_5"]["recognizers"] == ["template_hash", "ocr"]
    assert payload["metadata"]["augments"]["crop_path"] == payload["crops"]["augments"]


def test_analyze_frame_reports_unknown_profile(tmp_path: Path) -> None:
    screenshot = tmp_path / "tft.png"
    Image.new("RGB", (1920, 1080), color=(10, 20, 30)).save(screenshot)

    payload = analyze_frame("missing", screenshot)

    assert payload["success"] is False
    assert payload["error"]["code"] == "unknown_profile"


def test_analyze_frame_reports_missing_image(tmp_path: Path) -> None:
    payload = analyze_frame("tft", tmp_path / "missing.png")

    assert payload["success"] is False
    assert payload["error"]["code"] == "image_not_found"


def test_analyze_frame_reports_decode_failure(tmp_path: Path) -> None:
    bad_image = tmp_path / "bad.png"
    bad_image.write_text("not an image", encoding="utf-8")

    payload = analyze_frame("tft", bad_image)

    assert payload["success"] is False
    assert payload["error"]["code"] == "image_decode_failed"


def test_recognition_returns_unknown_without_templates(tmp_path: Path) -> None:
    screenshot = tmp_path / "tft.png"
    Image.new("RGB", (1920, 1080), color=(10, 20, 30)).save(screenshot)
    regions = grouped_screen_region_bboxes(1920, 1080)

    payload = recognize_shop_units(screenshot, regions, data_dir=tmp_path)

    assert payload["available"] is False
    assert len(payload["results"]) == 5
    assert {result["id"] for result in payload["results"]} == {"unknown"}


def test_tft_state_analysis_uses_mock_units_and_items() -> None:
    state = parse_tft_state(
        {
            "level": 6,
            "gold": 42,
            "stage": "3-2",
            "board_units": [
                {
                    "unit_id": "unit_a",
                    "star": 1,
                    "items": ["Needlessly Large Rod"],
                    "traits": ["mage", "backline"],
                },
                {"unit_id": "unit_b", "traits": ["mage", "frontline"]},
            ],
            "bench_units": [{"unit_id": "unit_a"}, {"unit_id": "unit_a"}],
            "shop_units": [],
            "items": ["Tear of the Goddess"],
        },
        {
            "mage": {"name": "mage", "tiers": [2, 4, 6], "units": ["unit_a", "unit_b"]},
            "backline": {"name": "backline", "tiers": [2, 4], "units": ["unit_a"]},
        },
    )

    assert [(trait.trait, trait.count, trait.active_tier, trait.next_tier) for trait in state.active_traits] == [
        ("mage", 2, 2, 4)
    ]
    assert any(gap.trait == "backline" and gap.units_needed == 1 for gap in state.trait_gaps)
    assert state.item_biases[0].bias in {"ap_carry", "mana_caster"}
    assert any(opportunity.champion_name == "unit_a" and opportunity.kind == "upgrade" for opportunity in state.upgrade_opportunities)


def test_insights_are_non_directive() -> None:
    insights = generate_insights(
        {
            "state": {"level": 6},
            "active_traits": [{"trait_id": "mage", "count": 2, "tier": 2}],
            "trait_gaps": [{"trait_id": "mage", "current": 3, "target": 4, "gap": 1}],
            "item_direction": {"direction": "ap", "scores": {"ap": 2}},
            "pairs": [{"unit_id": "unit_a", "count": 2}],
        }
    )

    assert {insight["type"] for insight in insights} >= {
        "trait_gap",
        "item_direction",
        "pairs",
        "composition_direction",
        "level_option",
    }
    rendered = " ".join(f"{insight['title']} {insight['detail']}" for insight in insights)
    assert not any(term in rendered for term in FORBIDDEN_DIRECTIVE_TERMS)
