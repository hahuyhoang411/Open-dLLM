from staff_simulation import StaffGenerator, audit_profile


def test_generated_profiles_are_valid() -> None:
  generator = StaffGenerator(seed=123)
  profiles = generator.generate_many(120)
  invalid_count = 0
  for profile in profiles:
    if not profile.validate().is_valid:
      invalid_count += 1
  assert invalid_count == 0


def test_dry_audit_runs_and_finds_expected_ranges() -> None:
  generator = StaffGenerator(seed=7)
  profiles = generator.generate_many(50)
  for profile in profiles:
    findings = audit_profile(profile)
    for finding in findings:
      assert finding.severity in {'warn', 'error'}
      assert finding.code
      if finding.severity == 'error':
        raise AssertionError(f'Unexpected hard validation error in audit: {finding.code}')
    assert 0 <= profile.burnout_score <= 1
    assert 0 <= profile.protocol_adherence_score <= 1
    assert profile.monthly_hours >= 80
