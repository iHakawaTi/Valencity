"""
HTML reporting module for valencity.
"""

from datetime import datetime
from pathlib import Path

from jinja2 import BaseLoader, Environment

from valencity import __version__
from valencity.pii import PIIReport
from valencity.privacy import ComplianceReport
from valencity.validation import ProfileReport, QualityReport

# Base CSS and layout
BASE_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>valencity Report - {{ title }}</title>
    <style>
        :root {
            --primary: #2563eb;
            --danger: #dc2626;
            --success: #16a34a;
            --warning: #ca8a04;
            --bg: #f8fafc;
            --card: #ffffff;
            --text: #1e293b;
            --border: #e2e8f0;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            line-height: 1.5;
            margin: 0;
            padding: 2rem;
        }
        
        .container {
            max-width: 1000px;
            margin: 0 auto;
        }
        
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border);
        }
        
        .logo {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--primary);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .meta {
            color: #64748b;
            font-size: 0.875rem;
        }
        
        .card {
            background: var(--card);
            border-radius: 0.5rem;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }
        
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }
        
        h2 { margin: 0; font-size: 1.25rem; }
        
        .badge {
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        
        .badge-danger { background: #fee2e2; color: var(--danger); }
        .badge-success { background: #dcfce7; color: var(--success); }
        .badge-warning { background: #fef9c3; color: var(--warning); }
        .badge-info { background: #e0f2fe; color: var(--primary); }
        
        table {
            width: 100%;
            border-collapse: collapse;
        }
        
        th, td {
            text-align: left;
            padding: 0.75rem;
            border-bottom: 1px solid var(--border);
        }
        
        th {
            font-weight: 600;
            color: #64748b;
            font-size: 0.875rem;
        }
        
        .progress-bar {
            width: 100%;
            height: 0.5rem;
            background: #e2e8f0;
            border-radius: 9999px;
            overflow: hidden;
        }
        
        .progress-value {
            height: 100%;
            background: var(--primary);
        }
        
        .footer {
            text-align: center;
            margin-top: 3rem;
            color: #94a3b8;
            font-size: 0.875rem;
        }
        
        .stat-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
        }
        
        .stat-item {
            display: flex;
            flex-direction: column;
        }
        
        .stat-label {
            font-size: 0.875rem;
            color: #64748b;
        }
        
        .stat-value {
            font-size: 1.25rem;
            font-weight: 600;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="logo">
                <span>🛡️ valencity</span>
            </div>
            <div class="meta">
                Generated: {{ generated_at }}<br>
                Version: {{ version }}
            </div>
        </div>
        
        <h1>{{ title }}</h1>
        
        {% block content %}{% endblock %}
        
        <div class="footer">
            Generated with valencity v{{ version }} • The ML Safety Fortress
        </div>
    </div>
</body>
</html>
"""

# PII Report Template
PII_TEMPLATE = """
{% extends "base" %}

{% block content %}
<div class="card">
    <div class="card-header">
        <h2>Summary</h2>
        {% if has_pii %}
        <span class="badge badge-danger">PII Detected</span>
        {% else %}
        <span class="badge badge-success">Safe</span>
        {% endif %}
    </div>
    <div class="stat-grid">
        <div class="stat-item">
            <span class="stat-label">Columns with PII</span>
            <span class="stat-value">{{ pii_columns_count }}</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Total Matches</span>
            <span class="stat-value">{{ total_matches }}</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Scanned Columns</span>
            <span class="stat-value">{{ scanned_columns }}</span>
        </div>
    </div>
</div>

<div class="card">
    <h2>Detailed Findings</h2>
    {% if columns %}
    <table>
        <thead>
            <tr>
                <th>Column</th>
                <th>PII Types</th>
                <th>Matches</th>
                <th>% of Rows</th>
                <th>Risk Level</th>
            </tr>
        </thead>
        <tbody>
            {% for col in columns %}
            <tr>
                <td style="font-weight: 500;">{{ col.name }}</td>
                <td>
                    {% for type in col.types %}
                    <span style="background: #e0e7ff; color: #3730a3; padding: 2px 6px; border-radius: 4px; font-size: 0.75rem;">
                        {{ type }}
                    </span>
                    {% endfor %}
                </td>
                <td>{{ col.match_count }}</td>
                <td>
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <div class="progress-bar" style="width: 60px;">
                            <div class="progress-value" style="width: {{ col.percentage }}%; background: {{ col.color }};"></div>
                        </div>
                        <span>{{ "%.1f"|format(col.percentage) }}%</span>
                    </div>
                </td>
                <td>
                    <span style="color: {{ col.color }}; font-weight: 600;">{{ col.risk }}</span>
                </td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
    {% else %}
    <p style="text-align: center; color: #64748b; padding: 2rem;">No PII detected in this dataset. Great job!</p>
    {% endif %}
</div>
{% endblock %}
"""

# Quality Report Template
QUALITY_TEMPLATE = """
{% extends "base" %}

{% block content %}
<div class="card">
    <div class="card-header">
        <h2>Summary</h2>
        {% if passed %}
        <span class="badge badge-success">Passed</span>
        {% else %}
        <span class="badge badge-danger">Failed</span>
        {% endif %}
    </div>
    <div class="stat-grid">
        <div class="stat-item">
            <span class="stat-label">Total Rows</span>
            <span class="stat-value">{{ total_rows }}</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Columns</span>
            <span class="stat-value">{{ total_columns }}</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Passed Checks</span>
            <span class="stat-value" style="color: var(--success);">{{ passed_count }}</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Failed Checks</span>
            <span class="stat-value" style="color: var(--danger);">{{ failed_count }}</span>
        </div>
    </div>
</div>

{% if failed_checks %}
<div class="card">
    <h2 style="color: var(--danger);">❌ Failed Checks</h2>
    <table>
        <thead>
            <tr>
                <th>Column</th>
                <th>Check Type</th>
                <th>Status</th>
                <th>Message</th>
            </tr>
        </thead>
        <tbody>
            {% for check in failed_checks %}
            <tr>
                <td style="font-weight: 500;">{{ check.column }}</td>
                <td>{{ check.type }}</td>
                <td><span class="badge badge-danger">FAIL</span></td>
                <td>{{ check.message }}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
</div>
{% endif %}

<div class="card">
    <h2>All Checks</h2>
    <table>
        <thead>
            <tr>
                <th>Column</th>
                <th>Check Type</th>
                <th>Status</th>
                <th>Metric Value</th>
            </tr>
        </thead>
        <tbody>
            {% for check in all_checks %}
            <tr>
                <td style="font-weight: 500;">{{ check.column }}</td>
                <td>{{ check.type }}</td>
                <td>
                    {% if check.status == "pass" %}
                    <span class="badge badge-success">PASS</span>
                    {% elif check.status == "warn" %}
                    <span class="badge badge-warning">WARN</span>
                    {% else %}
                    <span class="badge badge-danger">FAIL</span>
                    {% endif %}
                </td>
                <td>{{ check.value }}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
</div>
{% endblock %}
"""

# Profile Report Template
PROFILE_TEMPLATE = """
{% extends "base" %}

{% block content %}
<div class="card">
    <div class="card-header">
        <h2>Dataset Overview</h2>
        <span class="badge badge-info">Profile</span>
    </div>
    <div class="stat-grid">
        <div class="stat-item">
            <span class="stat-label">Total Rows</span>
            <span class="stat-value">{{ total_rows }}</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Total Columns</span>
            <span class="stat-value">{{ total_columns }}</span>
        </div>
    </div>
</div>

<div class="card">
    <h2>Column Statistics</h2>
    <table>
        <thead>
            <tr>
                <th>Column</th>
                <th>Type</th>
                <th>Missing</th>
                <th>Unique</th>
                <th>Stats (Mean/Min/Max)</th>
                <th>Distribution</th>
            </tr>
        </thead>
        <tbody>
            {% for col in columns %}
            <tr>
                <td style="font-weight: 600;">{{ col.name }}</td>
                <td><span class="badge badge-info">{{ col.dtype }}</span></td>
                <td>
                    {{ col.missing_count }}
                    <span style="color: #64748b; font-size: 0.8em;">({{ "%.1f"|format(col.missing_percentage) }}%)</span>
                </td>
                <td>
                    {{ col.unique_count }}
                    <span style="color: #64748b; font-size: 0.8em;">({{ "%.1f"|format(col.unique_percentage) }}%)</span>
                </td>
                <td style="font-size: 0.9em;">
                    {% if col.mean is not none %}
                    <div>μ: {{ "%.2f"|format(col.mean) }}</div>
                    <div style="color: #64748b;">[{{ "%.2f"|format(col.min_val) }} ... {{ "%.2f"|format(col.max_val) }}]</div>
                    {% else %}
                    <span style="color: #cbd5e1;">-</span>
                    {% endif %}
                </td>
                <td>
                    {% if col.mean is not none %}
                    {{ col.distribution_type }}
                    {% else %}
                    <span style="color: #cbd5e1;">-</span>
                    {% endif %}
                </td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
</div>
{% endblock %}
"""

# Compliance Report Template
COMPLIANCE_TEMPLATE = """
{% extends "base" %}

{% block content %}
<div class="card">
    <div class="card-header">
        <h2>Compliance Check</h2>
        {% if satisfied %}
        <span class="badge badge-success">Compliant</span>
        {% else %}
        <span class="badge badge-danger">Violations Found</span>
        {% endif %}
    </div>
    {% if satisfied %}
    <p style="text-align: center; color: var(--success); padding: 1rem;">
        No automated compliance violations detected.
    </p>
    {% else %}
    <div class="stat-grid">
        <div class="stat-item">
            <span class="stat-label">Total Violations</span>
            <span class="stat-value" style="color: var(--danger);">{{ violations|length }}</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Action Required</span>
            <span class="stat-value">Immediate</span>
        </div>
    </div>
    {% endif %}
</div>

{% if not satisfied %}
<div class="card">
    <h2 style="color: var(--danger);">Violations Detail</h2>
    <table>
        <thead>
            <tr>
                <th style="width: 20%;">Rule</th>
                <th style="width: 10%;">Severity</th>
                <th>Description</th>
            </tr>
        </thead>
        <tbody>
            {% for v in violations %}
            <tr>
                <td style="font-weight: 500;">{{ v.rule }}</td>
                <td>
                    {% if v.severity == "High" %}
                    <span class="badge badge-danger">High</span>
                    {% elif v.severity == "Medium" %}
                    <span class="badge badge-warning">Medium</span>
                    {% else %}
                    <span class="badge badge-info">Low</span>
                    {% endif %}
                </td>
                <td>{{ v.description }}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
</div>
{% endif %}
{% endblock %}
"""

class HTMLGenerator:
    """Generate HTML reports for valencity."""
    
    def __init__(self):
        self.env = Environment(loader=BaseLoader())
        self.env.loader.get_source = self._get_template_source
        
    def _get_template_source(self, environment, template):
        if template == "base":
            return BASE_TEMPLATE, "base", False
        elif template == "pii":
            return PII_TEMPLATE, "pii", False
        elif template == "quality":
            return QUALITY_TEMPLATE, "quality", False
        elif template == "profile":
            return PROFILE_TEMPLATE, "profile", False
        elif template == "compliance":
            return COMPLIANCE_TEMPLATE, "compliance", False
        return "", template, False

    def render_pii_report(self, report: PIIReport, output_path: Path):
        """Render a PII detection report to HTML."""
        template = self.env.get_template("pii")
        
        # Prepare context data
        columns_data = []
        for col_name, col_report in report.columns_with_pii.items():
            pct = col_report.pii_percentage
            risk = "High" if pct > 10 or any(t.value in ("ssn", "credit_card", "password") for t in col_report.pii_types_found) else "Medium"
            color = "#dc2626" if risk == "High" else "#ca8a04"
            
            columns_data.append({
                "name": col_name,
                "types": [t.value for t in col_report.pii_types_found],
                "match_count": col_report.match_count,
                "percentage": pct,
                "risk": risk,
                "color": color
            })
            
        context = {
            "title": "PII Scan Report",
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "version": __version__,
            "has_pii": report.has_pii,
            "pii_columns_count": len(report.pii_columns),
            "total_matches": report.total_matches,
            "scanned_columns": report.scanned_columns,
            "columns": columns_data
        }
        
        html = template.render(**context)
        output_path.write_text(html, encoding="utf-8")

    def render_quality_report(self, report: QualityReport, output_path: Path):
        """Render a Data Quality report to HTML."""
        template = self.env.get_template("quality")
        
        all_checks = []
        for check in report.checks:
            all_checks.append({
                "column": check.column or "Global",
                "type": check.check_type.value,
                "status": check.status.value,
                "value": f"{check.metric_value:.3f}" if isinstance(check.metric_value, float) else str(check.metric_value),
                "message": check.message
            })
            
        failed_checks = [c for c in all_checks if c["status"] == "fail"]
        
        context = {
            "title": "Data Quality Report",
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "version": __version__,
            "passed": report.passed,
            "total_rows": report.total_rows,
            "total_columns": report.total_columns,
            "passed_count": len([c for c in all_checks if c["status"] == "pass"]),
            "failed_count": len(failed_checks),
            "all_checks": all_checks,
            "failed_checks": failed_checks
        }
        
        html = template.render(**context)
        output_path.write_text(html, encoding="utf-8")
        
    def render_profile_report(self, report: ProfileReport, output_path: Path):
        """Render a Data Profile report to HTML."""
        template = self.env.get_template("profile")
        
        columns = []
        for col in report.columns.values():
            columns.append(col.to_dict())
            
        context = {
            "title": "Data Profile Report",
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "version": __version__,
            "total_rows": report.total_rows,
            "total_columns": report.total_columns,
            "columns": columns
        }
        
        html = template.render(**context)
        output_path.write_text(html, encoding="utf-8")
        
    def render_compliance_report(self, report: ComplianceReport, output_path: Path):
        """Render a Compliance report to HTML."""
        template = self.env.get_template("compliance")
        
        context = {
            "title": "Privay Compliance Report",
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "version": __version__,
            "satisfied": report.satisfied,
            "violations": [
                {"rule": v.rule, "severity": v.severity, "description": v.description}
                for v in report.violations
            ]
        }
        
        html = template.render(**context)
        output_path.write_text(html, encoding="utf-8")
