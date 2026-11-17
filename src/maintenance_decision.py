"""
Maintenance Decision Support System for EV Battery Health Monitoring
Maps predicted SOH, RUL, degradation category, and failure risk to actionable maintenance recommendations.
"""
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np


class DegradationSeverity(Enum):
    NORMAL = "Normal"
    ACCELERATED = "Accelerated"
    CRITICAL = "Critical"


class FailureRisk(Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class MaintenanceAction(Enum):
    NO_ACTION = "No Action Required"
    MONITOR = "Enhanced Monitoring"
    SCHEDULE_MAINTENANCE = "Schedule Maintenance"
    URGENT_MAINTENANCE = "Urgent Maintenance Required"
    REPLACE_BATTERY = "Battery Replacement Recommended"
    SAFETY_INSPECTION = "Safety Inspection Required"
    IMMEDIATE_REPLACEMENT = "Immediate Replacement Required"


@dataclass
class MaintenanceRecommendation:
    """Structured maintenance recommendation."""
    action: MaintenanceAction
    priority: str  # "Low", "Medium", "High", "Critical"
    description: str
    rationale: List[str]
    timeline: str  # e.g., "Within 1 week", "Within 1 month", "Next service interval"
    estimated_cost: Optional[str] = None
    safety_concern: bool = False


class MaintenanceDecisionEngine:
    """
    Rule-based decision engine for EV battery maintenance.
    Separates gradual degradation warnings from confirmed hardware faults/safety hazards.
    """
    
    # SOH thresholds (percentage)
    SOH_HEALTHY = 90
    SOH_DEGRADED = 80
    SOH_EOL = 70  # End of Life threshold
    SOH_CRITICAL = 60
    
    # RUL thresholds (cycles)
    RUL_PLENTY = 200
    RUL_MODERATE = 100
    RUL_LOW = 50
    RUL_CRITICAL = 20
    
    # Temperature thresholds (°C)
    TEMP_NORMAL_MAX = 40
    TEMP_ELEVATED = 50
    TEMP_HIGH = 60
    TEMP_CRITICAL = 70
    
    def __init__(self):
        self.rules = self._define_rules()
    
    def _define_rules(self) -> List[Dict]:
        """Define the rule set for maintenance decisions."""
        return [
            # CRITICAL SAFETY RULES (Highest priority - safety hazards)
            {
                'name': 'thermal_runaway_risk',
                'condition': lambda ctx: ctx.get('temp_max', 0) >= self.TEMP_CRITICAL,
                'action': MaintenanceAction.IMMEDIATE_REPLACEMENT,
                'priority': 'Critical',
                'description': 'CRITICAL: Battery temperature exceeds 70°C - Thermal runaway risk',
                'rationale': ['Temperature > 70°C indicates potential thermal runaway', 'Immediate safety hazard'],
                'timeline': 'IMMEDIATE - Stop vehicle operation',
                'safety_concern': True
            },
            {
                'name': 'critical_temperature',
                'condition': lambda ctx: ctx.get('temp_max', 0) >= self.TEMP_HIGH,
                'action': MaintenanceAction.SAFETY_INSPECTION,
                'priority': 'Critical',
                'description': 'CRITICAL: Battery temperature > 60°C - Requires immediate safety inspection',
                'rationale': ['High temperature accelerates degradation', 'Risk of thermal event', 'Check cooling system'],
                'timeline': 'Within 24 hours',
                'safety_concern': True
            },
            {
                'name': 'soh_critical_safety',
                'condition': lambda ctx: ctx.get('soh', 100) <= self.SOH_CRITICAL,
                'action': MaintenanceAction.IMMEDIATE_REPLACEMENT,
                'priority': 'Critical',
                'description': 'CRITICAL: SOH ≤ 60% - Battery below safe operating threshold',
                'rationale': ['Capacity below 60% indicates severe degradation', 'Risk of sudden failure', 'Cannot guarantee safe operation'],
                'timeline': 'IMMEDIATE - Do not operate vehicle',
                'safety_concern': True
            },
            
            # HIGH PRIORITY - Urgent maintenance
            {
                'name': 'rul_critical',
                'condition': lambda ctx: ctx.get('rul', 999) <= self.RUL_CRITICAL and ctx.get('eol_observed', False),
                'action': MaintenanceAction.REPLACE_BATTERY,
                'priority': 'High',
                'description': 'RUL ≤ 20 cycles with confirmed EOL trajectory - Battery replacement required',
                'rationale': ['Very few cycles remaining before EOL', 'EOL trajectory confirmed in data'],
                'timeline': 'Within 1 week'
            },
            {
                'name': 'degradation_critical_rul_low',
                'condition': lambda ctx: (
                    ctx.get('degradation_severity') == DegradationSeverity.CRITICAL.value 
                    and ctx.get('rul', 999) <= self.RUL_LOW
                ),
                'action': MaintenanceAction.URGENT_MAINTENANCE,
                'priority': 'High',
                'description': 'Critical degradation pattern with RUL ≤ 50 cycles - Urgent maintenance required',
                'rationale': ['Accelerated degradation detected', 'Low remaining useful life'],
                'timeline': 'Within 1 week'
            },
            {
                'name': 'high_failure_risk_soh_low',
                'condition': lambda ctx: (
                    ctx.get('failure_risk') == FailureRisk.HIGH.value 
                    and ctx.get('soh', 100) <= self.SOH_DEGRADED
                ),
                'action': MaintenanceAction.URGENT_MAINTENANCE,
                'priority': 'High',
                'description': 'High failure risk with SOH ≤ 80% - Urgent maintenance required',
                'rationale': ['High failure probability', 'Significant capacity loss'],
                'timeline': 'Within 1 week'
            },
            
            # MEDIUM PRIORITY - Schedule maintenance
            {
                'name': 'eol_approaching',
                'condition': lambda ctx: ctx.get('soh', 100) <= self.SOH_EOL,
                'action': MaintenanceAction.REPLACE_BATTERY,
                'priority': 'Medium',
                'description': 'SOH ≤ 70% (EOL threshold) - Battery replacement recommended',
                'rationale': ['Reached End-of-Life threshold', 'Capacity below 70% of rated'],
                'timeline': 'Within 1 month'
            },
            {
                'name': 'accelerated_degradation_rul_moderate',
                'condition': lambda ctx: (
                    ctx.get('degradation_severity') == DegradationSeverity.ACCELERATED.value
                    and ctx.get('rul', 999) <= self.RUL_MODERATE
                ),
                'action': MaintenanceAction.SCHEDULE_MAINTENANCE,
                'priority': 'Medium',
                'description': 'Accelerated degradation with RUL ≤ 100 cycles - Schedule maintenance',
                'rationale': ['Faster than normal degradation', 'Moderate remaining life'],
                'timeline': 'Within 2 weeks'
            },
            {
                'name': 'medium_failure_risk',
                'condition': lambda ctx: ctx.get('failure_risk') == FailureRisk.MEDIUM.value,
                'action': MaintenanceAction.SCHEDULE_MAINTENANCE,
                'priority': 'Medium',
                'description': 'Medium failure risk detected - Schedule preventive maintenance',
                'rationale': ['Elevated failure probability', 'Preventive action recommended'],
                'timeline': 'Next service interval or within 1 month'
            },
            {
                'name': 'elevated_temperature',
                'condition': lambda ctx: ctx.get('temp_max', 0) >= self.TEMP_ELEVATED,
                'action': MaintenanceAction.SCHEDULE_MAINTENANCE,
                'priority': 'Medium',
                'description': 'Battery temperature > 50°C - Check cooling system and schedule maintenance',
                'rationale': ['Elevated temperature accelerates aging', 'Cooling system may need service'],
                'timeline': 'Next service interval'
            },
            
            # LOW PRIORITY - Enhanced monitoring
            {
                'name': 'soh_degraded',
                'condition': lambda ctx: ctx.get('soh', 100) <= self.SOH_DEGRADED,
                'action': MaintenanceAction.MONITOR,
                'priority': 'Low',
                'description': 'SOH ≤ 80% - Enhanced monitoring recommended',
                'rationale': ['Capacity degradation detected', 'Track degradation rate'],
                'timeline': 'Monitor at each charge cycle'
            },
            {
                'name': 'accelerated_degradation',
                'condition': lambda ctx: ctx.get('degradation_severity') == DegradationSeverity.ACCELERATED.value,
                'action': MaintenanceAction.MONITOR,
                'priority': 'Low',
                'description': 'Accelerated degradation pattern detected - Increase monitoring frequency',
                'rationale': ['Degradation rate above normal', 'May indicate developing issue'],
                'timeline': 'Weekly monitoring'
            },
            {
                'name': 'low_failure_risk_early_warning',
                'condition': lambda ctx: (
                    ctx.get('failure_risk') == FailureRisk.LOW.value
                    and ctx.get('soh', 100) < self.SOH_HEALTHY
                ),
                'action': MaintenanceAction.MONITOR,
                'priority': 'Low',
                'description': 'Early degradation signs with low failure risk - Routine monitoring',
                'rationale': ['SOH below 90% but low immediate risk', 'Track for changes'],
                'timeline': 'Monthly monitoring'
            },
            
            # DEFAULT - No action
            {
                'name': 'healthy',
                'condition': lambda ctx: True,  # Catch-all
                'action': MaintenanceAction.NO_ACTION,
                'priority': 'Low',
                'description': 'Battery operating within normal parameters',
                'rationale': ['SOH > 90%', 'Normal degradation rate', 'Low failure risk', 'Temperature normal'],
                'timeline': 'Routine maintenance schedule'
            }
        ]
    
    def evaluate(self, context: Dict) -> List[MaintenanceRecommendation]:
        """
        Evaluate maintenance rules against context.
        Returns list of applicable recommendations (highest priority first).
        """
        recommendations = []
        
        for rule in self.rules:
            try:
                if rule['condition'](context):
                    rec = MaintenanceRecommendation(
                        action=rule['action'],
                        priority=rule['priority'],
                        description=rule['description'],
                        rationale=rule['rationale'],
                        timeline=rule['timeline'],
                        safety_concern=rule.get('safety_concern', False)
                    )
                    recommendations.append(rec)
                    
                    # Stop at first safety-critical rule
                    if rule.get('safety_concern', False):
                        break
                        
            except Exception as e:
                print(f"Rule evaluation error ({rule['name']}): {e}")
                continue
        
        # Sort by priority
        priority_order = {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3}
        recommendations.sort(key=lambda r: priority_order.get(r.priority, 4))
        
        return recommendations
    
    def get_primary_recommendation(self, context: Dict) -> MaintenanceRecommendation:
        """Get the highest priority recommendation."""
        recs = self.evaluate(context)
        return recs[0] if recs else MaintenanceRecommendation(
            action=MaintenanceAction.NO_ACTION,
            priority='Low',
            description='No specific recommendation',
            rationale=[],
            timeline='N/A'
        )


def create_maintenance_context(
    soh: float,
    rul: float,
    degradation_severity: str,
    failure_risk: str,
    temp_max: float,
    temp_mean: float,
    voltage_min: float,
    current_max: float,
    eol_observed: bool = False,
    cycle: int = 0
) -> Dict:
    """Create context dictionary for decision engine."""
    return {
        'soh': soh,
        'rul': rul,
        'degradation_severity': degradation_severity,
        'failure_risk': failure_risk,
        'temp_max': temp_max,
        'temp_mean': temp_mean,
        'voltage_min': voltage_min,
        'current_max': current_max,
        'eol_observed': eol_observed,
        'cycle': cycle
    }


def format_recommendation(rec: MaintenanceRecommendation) -> Dict:
    """Format recommendation for display."""
    return {
        'action': rec.action.value,
        'priority': rec.priority,
        'description': rec.description,
        'rationale': rec.rationale,
        'timeline': rec.timeline,
        'safety_concern': rec.safety_concern
    }


# Example usage and testing
if __name__ == "__main__":
    engine = MaintenanceDecisionEngine()
    
    # Test cases
    test_cases = [
        {
            'name': 'Healthy Battery',
            'context': create_maintenance_context(
                soh=95, rul=500, degradation_severity='Normal', failure_risk='Low',
                temp_max=35, temp_mean=30, voltage_min=3.2, current_max=1.5
            )
        },
        {
            'name': 'Early Degradation',
            'context': create_maintenance_context(
                soh=85, rul=200, degradation_severity='Normal', failure_risk='Low',
                temp_max=38, temp_mean=33, voltage_min=3.1, current_max=1.6
            )
        },
        {
            'name': 'Accelerated Degradation',
            'context': create_maintenance_context(
                soh=78, rul=80, degradation_severity='Accelerated', failure_risk='Medium',
                temp_max=45, temp_mean=40, voltage_min=3.0, current_max=1.8
            )
        },
        {
            'name': 'Critical Degradation',
            'context': create_maintenance_context(
                soh=65, rul=15, degradation_severity='Critical', failure_risk='High',
                temp_max=55, temp_mean=50, voltage_min=2.8, current_max=2.0,
                eol_observed=True
            )
        },
        {
            'name': 'Safety Hazard - High Temp',
            'context': create_maintenance_context(
                soh=75, rul=50, degradation_severity='Accelerated', failure_risk='High',
                temp_max=65, temp_mean=58, voltage_min=2.9, current_max=2.2
            )
        },
        {
            'name': 'Safety Hazard - Thermal Runaway Risk',
            'context': create_maintenance_context(
                soh=60, rul=5, degradation_severity='Critical', failure_risk='High',
                temp_max=75, temp_mean=70, voltage_min=2.5, current_max=3.0,
                eol_observed=True
            )
        }
    ]
    
    print("=" * 80)
    print("MAINTENANCE DECISION ENGINE - TEST CASES")
    print("=" * 80)
    
    for test in test_cases:
        print(f"\n{'='*80}")
        print(f"TEST: {test['name']}")
        print(f"{'='*80}")
        print(f"Context: SOH={test['context']['soh']}%, RUL={test['context']['rul']}, "
              f"Severity={test['context']['degradation_severity']}, "
              f"Risk={test['context']['failure_risk']}, "
              f"TempMax={test['context']['temp_max']}°C")
        
        recs = engine.evaluate(test['context'])
        
        for i, rec in enumerate(recs):
            print(f"\n  Recommendation {i+1} (Priority: {rec.priority}):")
            print(f"    Action: {rec.action.value}")
            print(f"    Description: {rec.description}")
            print(f"    Rationale: {'; '.join(rec.rationale)}")
            print(f"    Timeline: {rec.timeline}")
            print(f"    Safety Concern: {'YES' if rec.safety_concern else 'No'}")
            
            if rec.safety_concern:
                print("    ⚠️  SAFETY HAZARD DETECTED - IMMEDIATE ACTION REQUIRED")
                break