from scripts.verify_local import summarize


def test_unavailable_local_judge_cannot_become_a_quality_score():
    result = summarize([{"arm": "candidate", "status": "unavailable_diagnostics"}])
    assert result["arms"]["candidate"]["count_check_passes"] is None
    assert result["arms"]["candidate"]["unknown_images"] == 1
    assert result["full_contract_CSR"] is None


def test_count_support_does_not_become_full_csr_or_color_measurement():
    result = summarize([{"arm": "candidate", "status": "observed_diagnostics",
        "count_check_passed": True, "geometry_passed": True, "chromatic_binding_supported": None}])
    assert result["arms"]["candidate"]["count_check_passes"] == 1
    assert result["arms"]["candidate"]["chromatic_binding_supported_images"] is None
    assert result["promotion_status"] == "unqualified"
