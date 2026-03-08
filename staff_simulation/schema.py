from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional


ROLES = (
  'doctor',
  'resident',
  'nurse',
  'midwife',
  'pharmacist',
  'lab_tech',
  'imaging_tech',
  'community_health_worker',
  'admin_clinical',
)

FACILITY_LEVELS = ('central', 'provincial', 'district', 'commune', 'private')
REGIONS = (
  'red_river_delta',
  'northern_midlands',
  'north_central',
  'south_central',
  'central_highlands',
  'southeast',
  'mekong_delta',
)
LICENSE_TIERS = ('assistant', 'basic', 'advanced', 'specialist_i', 'specialist_ii')

DEPARTMENTS_BY_ROLE = {
  'doctor': ('internal_medicine', 'surgery', 'icu', 'obgyn', 'pediatrics', 'emergency', 'infectious_disease'),
  'resident': ('internal_medicine', 'surgery', 'icu', 'obgyn', 'pediatrics', 'emergency'),
  'nurse': ('internal_medicine', 'surgery', 'icu', 'obgyn', 'pediatrics', 'emergency', 'dialysis'),
  'midwife': ('obgyn', 'maternity', 'neonatal'),
  'pharmacist': ('pharmacy', 'oncology', 'critical_care'),
  'lab_tech': ('lab', 'microbiology', 'pathology', 'blood_bank'),
  'imaging_tech': ('radiology', 'ct_mri', 'ultrasound'),
  'community_health_worker': ('preventive_medicine', 'vaccination', 'maternal_child_health'),
  'admin_clinical': ('quality_management', 'infection_control', 'medical_records'),
}


@dataclass(frozen=True)
class ValidationIssue:
  code: str
  message: str


@dataclass(frozen=True)
class ValidationResult:
  issues: tuple[ValidationIssue, ...]

  @property
  def is_valid(self) -> bool:
    return len(self.issues) == 0


@dataclass(frozen=True)
class StaffProfile:
  staff_id: str
  region: str
  facility_level: str
  role: str
  department: str
  birth_year: int
  career_start_year: int
  specialty_start_year: int
  license_tier: str
  highest_degree: str
  monthly_hours: int
  night_shifts_per_month: int
  on_call_days_per_month: int
  has_secondary_job: bool
  secondary_job_hours: int
  secondary_job_type: Optional[str]
  public_sector_salary_index: float
  burnout_score: float
  protocol_adherence_score: float
  resource_constraint_exposure: float
  informal_hierarchy_pressure: float
  patient_gift_pressure: float
  party_membership: bool
  promotion_track_strength: float
  interpersonal_conflict_risk: float
  recent_med_error_incidents: int
  outdated_protocol_risk: float

  @property
  def age(self) -> int:
    return date.today().year - self.birth_year

  @property
  def years_experience(self) -> int:
    return date.today().year - self.career_start_year

  @property
  def years_in_specialty(self) -> int:
    return date.today().year - self.specialty_start_year

  @property
  def seniority_level(self) -> str:
    if self.years_experience >= 15:
      return 'expert'
    if self.years_experience >= 8:
      return 'senior'
    if self.years_experience >= 3:
      return 'intermediate'
    return 'junior'

  def validate(self) -> ValidationResult:
    issues: list[ValidationIssue] = []

    if self.role not in ROLES:
      issues.append(ValidationIssue('role.invalid', f'Unknown role: {self.role}'))
      return ValidationResult(tuple(issues))
    if self.facility_level not in FACILITY_LEVELS:
      issues.append(ValidationIssue('facility.invalid', f'Unknown facility level: {self.facility_level}'))
    if self.region not in REGIONS:
      issues.append(ValidationIssue('region.invalid', f'Unknown region: {self.region}'))
    if self.department not in DEPARTMENTS_BY_ROLE[self.role]:
      issues.append(
        ValidationIssue('department.role_mismatch', f'{self.department} incompatible with role {self.role}')
      )
    if self.license_tier not in LICENSE_TIERS:
      issues.append(ValidationIssue('license.invalid', f'Unknown license tier: {self.license_tier}'))

    if self.birth_year < 1955 or self.birth_year > date.today().year - 18:
      issues.append(ValidationIssue('age.out_of_bounds', 'Birth year out of workforce bounds'))
    if self.career_start_year < self.birth_year + 18:
      issues.append(ValidationIssue('career.too_early', 'Career start year is too early'))
    if self.career_start_year > date.today().year:
      issues.append(ValidationIssue('career.in_future', 'Career start year cannot be in the future'))
    if self.specialty_start_year < self.career_start_year:
      issues.append(ValidationIssue('specialty.before_career', 'Specialty start cannot precede career start'))
    if self.specialty_start_year > date.today().year:
      issues.append(ValidationIssue('specialty.in_future', 'Specialty start year cannot be in the future'))
    if self.years_in_specialty > self.years_experience:
      issues.append(ValidationIssue('experience.specialty_exceeds_total', 'Specialty years exceed total experience'))
    if self.years_experience > self.age - 18:
      issues.append(ValidationIssue('experience.age_impossible', 'Experience exceeds plausible years since adulthood'))

    role_license_floor = {
      'resident': 'basic',
      'doctor': 'advanced',
      'nurse': 'basic',
      'midwife': 'basic',
      'pharmacist': 'advanced',
      'lab_tech': 'assistant',
      'imaging_tech': 'assistant',
      'community_health_worker': 'assistant',
      'admin_clinical': 'basic',
    }
    tier_rank = {tier: idx for idx, tier in enumerate(LICENSE_TIERS)}
    if self.license_tier in tier_rank and tier_rank[self.license_tier] < tier_rank[role_license_floor[self.role]]:
      issues.append(ValidationIssue('license.role_floor', f'{self.license_tier} below floor for {self.role}'))
    if self.role == 'resident' and self.years_experience > 8:
      issues.append(ValidationIssue('resident.tenure_unlikely', 'Resident tenure unusually long'))
    if self.role == 'doctor' and self.years_experience < 2:
      issues.append(ValidationIssue('doctor.experience_too_low', 'Doctor profile with too little experience'))
    if self.role == 'midwife' and self.department not in {'obgyn', 'maternity', 'neonatal'}:
      issues.append(ValidationIssue('midwife.department_invalid', 'Midwife outside maternal departments'))
    if self.facility_level == 'commune' and self.department in {'ct_mri', 'icu', 'pathology', 'critical_care'}:
      issues.append(ValidationIssue('facility.department_unavailable', 'Advanced unit unlikely at commune level'))

    if self.monthly_hours < 80 or self.monthly_hours > 340:
      issues.append(ValidationIssue('schedule.hours_out_of_bounds', 'Monthly hours out of plausible bounds'))
    if self.night_shifts_per_month < 0 or self.night_shifts_per_month > 20:
      issues.append(ValidationIssue('schedule.night_shift_out_of_bounds', 'Night shifts out of plausible bounds'))
    if self.on_call_days_per_month < 0 or self.on_call_days_per_month > 15:
      issues.append(ValidationIssue('schedule.on_call_out_of_bounds', 'On-call days out of plausible bounds'))
    if self.monthly_hours < self.night_shifts_per_month * 8:
      issues.append(ValidationIssue('schedule.hours_night_inconsistent', 'Night shift load exceeds monthly hours'))
    if self.has_secondary_job and self.secondary_job_hours <= 0:
      issues.append(ValidationIssue('moonlighting.hours_missing', 'Secondary job enabled but hours are zero'))
    if (not self.has_secondary_job) and self.secondary_job_hours > 0:
      issues.append(ValidationIssue('moonlighting.hours_without_job', 'Secondary hours without secondary job'))
    if self.has_secondary_job and self.secondary_job_type is None:
      issues.append(ValidationIssue('moonlighting.type_missing', 'Secondary job type missing'))
    if (not self.has_secondary_job) and self.secondary_job_type is not None:
      issues.append(ValidationIssue('moonlighting.type_without_job', 'Secondary job type set while disabled'))
    if self.secondary_job_hours > 90:
      issues.append(ValidationIssue('moonlighting.hours_extreme', 'Secondary job hours implausibly high'))

    score_fields = {
      'burnout_score': self.burnout_score,
      'protocol_adherence_score': self.protocol_adherence_score,
      'resource_constraint_exposure': self.resource_constraint_exposure,
      'informal_hierarchy_pressure': self.informal_hierarchy_pressure,
      'patient_gift_pressure': self.patient_gift_pressure,
      'promotion_track_strength': self.promotion_track_strength,
      'interpersonal_conflict_risk': self.interpersonal_conflict_risk,
      'outdated_protocol_risk': self.outdated_protocol_risk,
    }
    for name, value in score_fields.items():
      if value < 0.0 or value > 1.0:
        issues.append(ValidationIssue('score.out_of_bounds', f'{name} must be in [0, 1]'))
    if self.public_sector_salary_index <= 0:
      issues.append(ValidationIssue('salary.index_non_positive', 'Salary index must be positive'))
    if self.recent_med_error_incidents < 0 or self.recent_med_error_incidents > 12:
      issues.append(ValidationIssue('safety.error_incident_out_of_bounds', 'Incident count out of plausible range'))
    if self.protocol_adherence_score > 0.85 and self.outdated_protocol_risk > 0.6:
      issues.append(ValidationIssue('practice.protocol_conflict', 'High adherence with high outdated risk conflicts'))
    if self.monthly_hours > 270 and self.burnout_score < 0.2:
      issues.append(ValidationIssue('behavior.burnout_hours_mismatch', 'Extreme workload with very low burnout'))
    if self.burnout_score > 0.8 and self.interpersonal_conflict_risk < 0.15:
      issues.append(
        ValidationIssue('behavior.burnout_conflict_mismatch', 'Severe burnout with near-zero conflict risk')
      )

    return ValidationResult(tuple(issues))
