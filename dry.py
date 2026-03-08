from __future__ import annotations

import argparse
import json
from typing import Any

from staff_simulation import StaffGenerator, audit_profile, summarize_loops
from staff_simulation.schema import StaffProfile


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument('--loops', type=int, default=10)
  parser.add_argument('--sample-size', type=int, default=5)
  parser.add_argument('--seed', type=int, default=1337)
  parser.add_argument('--json', action='store_true')
  return parser.parse_args()


def compact_profile_view(profile: StaffProfile) -> dict[str, Any]:
  return {
    'staff_id': profile.staff_id,
    'region': profile.region,
    'facility_level': profile.facility_level,
    'role': profile.role,
    'department': profile.department,
    'age': profile.age,
    'years_experience': profile.years_experience,
    'years_in_specialty': profile.years_in_specialty,
    'license_tier': profile.license_tier,
    'seniority_level': profile.seniority_level,
    'monthly_hours': profile.monthly_hours,
    'night_shifts_per_month': profile.night_shifts_per_month,
    'has_secondary_job': profile.has_secondary_job,
    'secondary_job_hours': profile.secondary_job_hours,
    'burnout_score': profile.burnout_score,
    'resource_constraint_exposure': profile.resource_constraint_exposure,
    'informal_hierarchy_pressure': profile.informal_hierarchy_pressure,
    'patient_gift_pressure': profile.patient_gift_pressure,
    'recent_med_error_incidents': profile.recent_med_error_incidents,
    'outdated_protocol_risk': profile.outdated_protocol_risk,
  }


def stratified_sample(generator: StaffGenerator, sample_size: int) -> list[StaffProfile]:
  role_plan = ['doctor', 'nurse', 'midwife', 'pharmacist', 'resident']
  selected_roles = role_plan[:sample_size]
  while len(selected_roles) < sample_size:
    selected_roles.append('community_health_worker')
  return [generator.generate_one(role_override=role) for role in selected_roles]


def main() -> None:
  args = parse_args()
  generator = StaffGenerator(seed=args.seed)

  all_findings = []
  loop_outputs: list[dict[str, Any]] = []

  for loop_idx in range(1, args.loops + 1):
    profiles = stratified_sample(generator, args.sample_size)
    loop_findings = []
    rendered_profiles = []

    for profile in profiles:
      findings = audit_profile(profile)
      loop_findings.append(findings)
      all_findings.append(findings)
      rendered_profiles.append(compact_profile_view(profile))

    loop_summary = summarize_loops(loop_findings)
    loop_outputs.append({
      'loop': loop_idx,
      'profiles': rendered_profiles,
      'summary': loop_summary,
    })

  global_summary = summarize_loops(all_findings)

  payload = {
    'config': {'loops': args.loops, 'sample_size': args.sample_size, 'seed': args.seed},
    'global_summary': global_summary,
    'loops': loop_outputs,
  }

  if args.json:
    print(json.dumps(payload, indent=2))
    return

  print('=== Staff Simulation Dry Audit ===')
  print(f'Loops: {args.loops}, Sample per loop: {args.sample_size}, Seed: {args.seed}')
  print(f'Global findings: {payload["global_summary"]}')
  for loop in loop_outputs:
    print(f'\n--- Loop {loop["loop"]} ---')
    print(f'Summary: {loop["summary"]}')
    for profile in loop['profiles']:
      print(
        f'- {profile["staff_id"]} | {profile["facility_level"]} | {profile["role"]} | '
        f'{profile["department"]} | exp={profile["years_experience"]}y | '
        f'hours={profile["monthly_hours"]} | burnout={profile["burnout_score"]} | '
        f'moonlight={profile["has_secondary_job"]}:{profile["secondary_job_hours"]}'
      )


if __name__ == '__main__':
  main()
