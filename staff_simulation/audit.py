from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .schema import StaffProfile


@dataclass(frozen=True)
class AuditFinding:
  code: str
  severity: str
  message: str


def audit_profile(profile: StaffProfile) -> tuple[AuditFinding, ...]:
  findings: list[AuditFinding] = []

  validation = profile.validate()
  for issue in validation.issues:
    findings.append(AuditFinding(code=issue.code, severity='error', message=issue.message))

  if profile.monthly_hours >= 260 and profile.burnout_score < 0.35:
    findings.append(
      AuditFinding(
        code='realism.burnout_understated',
        severity='warn',
        message='Very high workload with low burnout score is uncommon.',
      )
    )
  if profile.has_secondary_job and profile.monthly_hours + profile.secondary_job_hours > 300:
    findings.append(
      AuditFinding(
        code='realism.extreme_total_hours',
        severity='warn',
        message='Combined primary and secondary workload exceeds 300h/month.',
      )
    )
  if profile.facility_level in {'district', 'commune'} and profile.outdated_protocol_risk < 0.1:
    findings.append(
      AuditFinding(
        code='realism.rural_protocol_too_modern',
        severity='warn',
        message='Very low outdated protocol risk in under-resourced settings may be optimistic.',
      )
    )
  if profile.role == 'resident' and profile.informal_hierarchy_pressure < 0.2:
    findings.append(
      AuditFinding(
        code='realism.resident_hierarchy_too_low',
        severity='warn',
        message='Resident hierarchy pressure is likely under-modeled.',
      )
    )
  if profile.burnout_score > 0.75 and profile.recent_med_error_incidents == 0:
    findings.append(
      AuditFinding(
        code='realism.burnout_without_incidents',
        severity='warn',
        message='High burnout with zero recent incidents may understate performance impact.',
      )
    )
  if profile.facility_level == 'commune' and profile.has_secondary_job and profile.secondary_job_hours > 30:
    findings.append(
      AuditFinding(
        code='realism.commune_moonlighting_strain',
        severity='warn',
        message='Heavy moonlighting in commune settings may strain availability.',
      )
    )
  if profile.years_experience < 6 and profile.license_tier == 'specialist_ii':
    findings.append(
      AuditFinding(
        code='realism.fast_track_specialist',
        severity='warn',
        message='Specialist II with very low experience should remain a rare edge case.',
      )
    )

  return tuple(findings)


def summarize_loops(loop_findings: list[tuple[AuditFinding, ...]]) -> dict[str, object]:
  severity_counter: Counter[str] = Counter()
  code_counter: Counter[str] = Counter()
  for findings in loop_findings:
    for finding in findings:
      severity_counter[finding.severity] += 1
      code_counter[finding.code] += 1
  return {
    'total_findings': sum(severity_counter.values()),
    'by_severity': dict(severity_counter),
    'top_codes': code_counter.most_common(10),
  }
