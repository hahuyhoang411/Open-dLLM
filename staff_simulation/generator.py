from __future__ import annotations

import random
import uuid
from dataclasses import dataclass
from datetime import date

from .schema import DEPARTMENTS_BY_ROLE, LICENSE_TIERS, REGIONS, StaffProfile


@dataclass(frozen=True)
class DistributionConfig:
  role_weights: dict[str, float]
  facility_weights: dict[str, float]


DEFAULT_DISTRIBUTION = DistributionConfig(
  role_weights={
    'nurse': 0.34,
    'doctor': 0.19,
    'resident': 0.06,
    'midwife': 0.07,
    'pharmacist': 0.08,
    'lab_tech': 0.09,
    'imaging_tech': 0.06,
    'community_health_worker': 0.07,
    'admin_clinical': 0.04,
  },
  facility_weights={
    'central': 0.09,
    'provincial': 0.24,
    'district': 0.3,
    'commune': 0.22,
    'private': 0.15,
  },
)


class StaffGenerator:
  def __init__(self, seed: int = 42, distribution: DistributionConfig = DEFAULT_DISTRIBUTION) -> None:
    self.rng = random.Random(seed)
    self.distribution = distribution

  def _pick(self, weighted_map: dict[str, float]) -> str:
    keys = list(weighted_map.keys())
    weights = list(weighted_map.values())
    return self.rng.choices(keys, weights=weights, k=1)[0]

  def _experience_for_role(self, role: str) -> tuple[int, int, int]:
    if role == 'resident':
      age = self.rng.randint(24, 34)
      exp = self.rng.randint(1, min(8, age - 23))
      specialty = self.rng.randint(0, exp)
      return age, exp, specialty
    if role in {'doctor', 'pharmacist'}:
      age = self.rng.randint(26, 62)
      exp = self.rng.randint(2, age - 24)
      specialty = self.rng.randint(0, exp)
      return age, exp, specialty
    if role in {'nurse', 'midwife'}:
      age = self.rng.randint(21, 60)
      exp = self.rng.randint(0, age - 20)
      specialty = self.rng.randint(0, exp)
      return age, exp, specialty
    if role in {'lab_tech', 'imaging_tech', 'community_health_worker'}:
      age = self.rng.randint(20, 58)
      exp = self.rng.randint(0, age - 19)
      specialty = self.rng.randint(0, exp)
      return age, exp, specialty
    age = self.rng.randint(22, 60)
    exp = self.rng.randint(0, age - 21)
    specialty = self.rng.randint(0, exp)
    return age, exp, specialty

  def _license_for_role(self, role: str, years_experience: int) -> str:
    floor = {
      'resident': 1,
      'doctor': 2,
      'nurse': 1,
      'midwife': 1,
      'pharmacist': 2,
      'lab_tech': 0,
      'imaging_tech': 0,
      'community_health_worker': 0,
      'admin_clinical': 1,
    }[role]
    cap = min(len(LICENSE_TIERS) - 1, floor + years_experience // 7)
    idx = self.rng.randint(floor, max(floor, cap))
    return LICENSE_TIERS[idx]

  def _degree_for_role(self, role: str) -> str:
    if role == 'resident':
      return self.rng.choice(['medical_doctor', 'medical_doctor', 'master_medicine'])
    if role == 'doctor':
      return self.rng.choice(['medical_doctor', 'medical_doctor', 'specialist_i', 'master_medicine', 'specialist_ii'])
    if role in {'nurse', 'midwife'}:
      return self.rng.choice(['college', 'bachelor', 'bachelor', 'master_nursing'])
    if role in {'pharmacist'}:
      return self.rng.choice(['bachelor_pharmacy', 'master_pharmacy', 'specialist_i'])
    return self.rng.choice(['college', 'bachelor'])

  def _hours_profile(self, role: str, facility_level: str, has_secondary_job: bool) -> tuple[int, int, int, int]:
    base_hours = {
      'resident': self.rng.randint(210, 300),
      'doctor': self.rng.randint(170, 270),
      'nurse': self.rng.randint(180, 280),
      'midwife': self.rng.randint(170, 260),
      'pharmacist': self.rng.randint(160, 230),
      'lab_tech': self.rng.randint(150, 220),
      'imaging_tech': self.rng.randint(150, 230),
      'community_health_worker': self.rng.randint(140, 220),
      'admin_clinical': self.rng.randint(150, 210),
    }[role]
    if facility_level in {'district', 'commune'}:
      base_hours += self.rng.randint(-5, 20)
    if facility_level == 'central':
      base_hours += self.rng.randint(0, 25)

    night_shifts = 0
    on_call = 0
    if role in {'resident', 'doctor', 'nurse', 'midwife'}:
      night_shifts = self.rng.randint(3, 14)
      on_call = self.rng.randint(1, 8)
    elif role in {'lab_tech', 'imaging_tech'}:
      night_shifts = self.rng.randint(1, 8)
      on_call = self.rng.randint(0, 4)

    secondary_hours = 0
    if has_secondary_job:
      secondary_hours = self.rng.randint(8, 48)

    return base_hours, night_shifts, on_call, secondary_hours

  def _secondary_job_type(self, role: str, has_secondary_job: bool) -> str | None:
    if not has_secondary_job:
      return None
    if role in {'doctor', 'resident', 'nurse', 'midwife'}:
      return self.rng.choice(['private_clinic', 'teleconsult', 'teaching', 'home_care'])
    return self.rng.choice(['teaching', 'pharmacy_shift', 'community_program'])

  def _score(self, low: float, high: float) -> float:
    return round(self.rng.uniform(low, high), 3)

  def _role_from_override(self, role: str | None) -> str:
    if role is not None:
      return role
    return self._pick(self.distribution.role_weights)

  def generate_one(self, role_override: str | None = None) -> StaffProfile:
    role = self._role_from_override(role_override)
    facility_level = self._pick(self.distribution.facility_weights)
    region = self.rng.choice(REGIONS)
    department = self.rng.choice(DEPARTMENTS_BY_ROLE[role])

    if facility_level == 'commune' and department in {'icu', 'critical_care', 'ct_mri', 'pathology'}:
      department = self.rng.choice(
        tuple(dep for dep in DEPARTMENTS_BY_ROLE[role] if dep not in {'icu', 'critical_care', 'ct_mri', 'pathology'})
      )

    age, years_experience, years_in_specialty = self._experience_for_role(role)
    current_year = date.today().year
    birth_year = current_year - age
    career_start_year = current_year - years_experience
    specialty_start_year = current_year - years_in_specialty
    license_tier = self._license_for_role(role, years_experience)
    highest_degree = self._degree_for_role(role)

    has_secondary_job = self.rng.random() < (0.42 if role in {'doctor', 'nurse', 'pharmacist', 'midwife'} else 0.25)
    monthly_hours, night_shifts, on_call, secondary_job_hours = self._hours_profile(
      role, facility_level, has_secondary_job
    )
    secondary_job_type = self._secondary_job_type(role, has_secondary_job)

    resource_pressure = self._score(0.55, 0.95) if facility_level in {'district', 'commune'} else self._score(0.2, 0.75)
    hierarchy_pressure = self._score(0.3, 0.9) if role in {'resident', 'nurse', 'lab_tech'} else self._score(0.2, 0.8)
    gift_pressure = self._score(0.2, 0.8) if role in {'doctor', 'nurse', 'midwife'} else self._score(0.05, 0.45)
    burnout_floor = min(0.9, (monthly_hours - 130) / 230)
    burnout = self._score(max(0.15, burnout_floor - 0.2), min(0.98, burnout_floor + 0.25))
    adherence = self._score(0.45, 0.95)
    conflict = self._score(0.1, 0.85)
    outdated_risk = self._score(0.25, 0.8) if years_experience > 20 else self._score(0.05, 0.55)

    med_error_base = 0
    if monthly_hours > 250:
      med_error_base += 1
    if burnout > 0.75:
      med_error_base += 1
    if facility_level in {'district', 'commune'} and resource_pressure > 0.8:
      med_error_base += 1
    recent_med_error_incidents = min(4, med_error_base + self.rng.randint(0, 1))

    party_membership = self.rng.random() < (0.35 if years_experience >= 8 else 0.12)
    promotion_track_strength = self._score(0.2, 0.9)
    if party_membership:
      promotion_track_strength = round(min(1.0, promotion_track_strength + self.rng.uniform(0.05, 0.2)), 3)

    salary_index = {
      'commune': self._score(0.75, 1.15),
      'district': self._score(0.85, 1.25),
      'provincial': self._score(0.95, 1.45),
      'central': self._score(1.1, 1.8),
      'private': self._score(1.15, 2.4),
    }[facility_level]

    profile = StaffProfile(
      staff_id=f'staff-{uuid.uuid4().hex[:10]}',
      region=region,
      facility_level=facility_level,
      role=role,
      department=department,
      birth_year=birth_year,
      career_start_year=career_start_year,
      specialty_start_year=specialty_start_year,
      license_tier=license_tier,
      highest_degree=highest_degree,
      monthly_hours=monthly_hours,
      night_shifts_per_month=night_shifts,
      on_call_days_per_month=on_call,
      has_secondary_job=has_secondary_job,
      secondary_job_hours=secondary_job_hours,
      secondary_job_type=secondary_job_type,
      public_sector_salary_index=salary_index,
      burnout_score=burnout,
      protocol_adherence_score=adherence,
      resource_constraint_exposure=resource_pressure,
      informal_hierarchy_pressure=hierarchy_pressure,
      patient_gift_pressure=gift_pressure,
      party_membership=party_membership,
      promotion_track_strength=promotion_track_strength,
      interpersonal_conflict_risk=conflict,
      recent_med_error_incidents=recent_med_error_incidents,
      outdated_protocol_risk=outdated_risk,
    )

    validation = profile.validate()
    if not validation.is_valid:
      return self.generate_one(role_override=role_override)
    return profile

  def generate_many(self, count: int) -> list[StaffProfile]:
    return [self.generate_one() for _ in range(count)]
