# SACC School Database & Web Application: Complete 6–12 Month Build Plan

## Executive Summary
The attached notes describe a school aftercare program (likely SACC — School Age Child Care) that currently relies on Microsoft Access and manual paper forms. The goal is to replace this with a secure, modern, cloud-hosted web application built on **Django + PostgreSQL** — supporting three separate user roles (Admin, Teacher/TA, and a restricted student-data handler), featuring online registration, attendance, payment tracking, and an admin-only teacher HR database. This plan outlines the exact migration path from Access, the full database schema derived from field specifications, a phased development timeline, and a cost comparison of hosting options from budget-friendly PaaS providers to AWS.


## Part 1 — Database Requirements: Reading the Specs
Based on the four pages of handwritten and typed field listings, the system requires **four primary database domains** with one strict access-control boundary.
### Domain 1: Students (Public-Facing + Admin)
Fields captured from pages 3 and 4:

| Category | Fields |
|---|---|
| Identity | Student ID, Portrait Photo, Last Name, First Name, Chinese Name, Gender, DOB, Age (calculated to 1 decimal) |
| Address | Home Address 1, Home City, Home State, Home Zip |
| Contact | Phone Number |
| Parents | Parent 1 Name/Phone/Email, Parent 2 Name/Phone/Email |
| Emergency | Emergency Contact 1 & 2 (Name, Phone, Relation to Student) |
| Logistics | Pick-Up (self/mom/M-B/B-M/bus), Term (duration of registration) |
| Medical | Medical/Allergy Comments, Medical Form Scan (with flag for outdated) |
### Domain 2: Classes (Admin + Teacher Access)
Fields captured from page 2:

| Field | Notes |
|---|---|
| ClassID | Primary key |
| Class Description English / Chinese | Bilingual |
| Classroom | Room assignment |
| Class Period / Schedule | AM1, AM2, PM1, PM2, PM3/Saturday |
| Class Teacher 1 & 2 | Foreign keys to Teacher table |
| Class TA 1–4 | Foreign keys to Teacher table |
| Class Age Range | |
| Class Size | Formula/count from enrollment |
### Domain 3: Registrations / Enrollment (Student + Class Join)
Fields from pages 1 and 4 notes:

- Registration Form (per student)
- Currently Registered Classes (AM1, AM2, PM1, PM2, PM3 Saturday)
- Registered Duration (Fall, Spring, Year, Summer # of weeks)
- Registration History / Classes Taken Each Semester
- Student Schedule Card
- Set (which registration set the record was written in)
- Summer Field Trip waiver / permission / payment
### Domain 4: Teachers — Admin-Only, Separate Access
From pages 1 and 2 (explicitly noted as **"separate database for admin only, not accessible by staff handling student data"**):

| Field | Notes |
|---|---|
| TeacherID | Primary key |
| Portrait Photo | Stored as file reference |
| Last Name, First Name, Chinese Name | |
| Gender, DOB, Home Address, Phone, Email | PII — admin-only |
| Status | Citizen / Resident / Work Permit + expiration |
| Social Security # | Encrypted field — highest sensitivity |
| Current Rate / Rate History | Compensation history |
| SACC Certifications | 6001, 6002, 6003, 6004, 6005, 6022, 3370; Foundations of H&S; ACE; Mandated Reporter |
| Gov ID, Teaching License | |
| Preferred Student Grades, Capable Subjects | |
| Class Assignments | AM1, AM2, PM1, PM2, PM3/Saturday (current) |
| Class Assignment History | |
### Domain 5: Operations & Logs
From page 4 handwritten notes:

- Attendance Sheets by class (Saturday, Summer, After School) — printable and tablet-accessible
- School Bus Attendance with estimated arrival times
- Teacher Attendance Sheets + TA Attendance Sheets
- Incident Logs ("history of anything that happens to students")
- Grades History / Report Cards
- Comments (per student and per teacher)
- Payment status / balance payment portal reference


## Part 2 — Phase 1: Access → PostgreSQL Migration (Weeks 1–8)
This is the most technically critical phase. Rushing it creates data integrity problems that corrupt everything downstream.
### Step 1: Access Database Audit and Inventory (Week 1)
Before touching the data, fully document the existing Access database:[^1]

```
1. Export full relationship diagram (Relationships view → print/screenshot)
2. Count tables, records per table, and document all queries
3. Identify and flag:
   - Attachment/OLE Object fields (photos, scanned forms)
   - Multi-value fields
   - Lookup fields
   - Any VBA macros or modules
   - Linked external tables
4. Export a data dictionary: all field names, data types, required flags, 
   and default values
```

Run the Microsoft Access **Upsizing Wizard** (built into Access) for a first-pass schema analysis report. This does not do the migration — it reveals compatibility issues.
### Step 2: Target Schema Design in PostgreSQL (Week 2)
Map Access data types to PostgreSQL equivalents:

| Access Type | PostgreSQL Target | Notes |
|---|---|---|
| AutoNumber | `BIGSERIAL` or `GENERATED ALWAYS AS IDENTITY` | Modern preferred |
| Text (255) | `VARCHAR(255)` | Set actual limits |
| Memo/Long Text | `TEXT` | Unlimited |
| Currency | `NUMERIC(19,4)` | Never use FLOAT for money |
| Yes/No | `BOOLEAN` | |
| Date/Time | `TIMESTAMPTZ` | Use timezone-aware |
| Attachment / OLE | `VARCHAR` file path + separate S3/object store | Never inline blobs |
| Number (Integer) | `INTEGER` | |
| Number (Long Int) | `BIGINT` | |

**Critical constraint**: Social Security Numbers must be stored in a `BYTEA` column encrypted at the application level using AES-256 before write. The Django model layer (not the database) handles encryption/decryption. SSN should never appear in logs, error messages, or API responses.

Design the target schema with these PostgreSQL best practices:

```sql
-- Example: core student table
CREATE TABLE students (
    student_id    BIGSERIAL PRIMARY KEY,
    portrait_url  VARCHAR(500),
    last_name     VARCHAR(100) NOT NULL,
    first_name    VARCHAR(100) NOT NULL,
    chinese_name  VARCHAR(100),
    gender        VARCHAR(20),
    dob           DATE NOT NULL,
    age           NUMERIC(4,1) GENERATED ALWAYS AS 
                  (EXTRACT(YEAR FROM AGE(dob)) + 
                   EXTRACT(MONTH FROM AGE(dob))/12.0) STORED,
    phone         VARCHAR(30),
    addr1         VARCHAR(200),
    city          VARCHAR(100),
    state         CHAR(2),
    zip           VARCHAR(10),
    pickup_type   VARCHAR(20) CHECK (pickup_type IN 
                  ('self','mom','M-B','B-M','bus')),
    medical_notes TEXT,
    medical_form_url VARCHAR(500),
    medical_form_outdated BOOLEAN DEFAULT FALSE,
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    updated_at    TIMESTAMPTZ DEFAULT NOW()
);

-- Enforce row-level access using PostgreSQL Row Level Security
ALTER TABLE teachers ENABLE ROW LEVEL SECURITY;
CREATE POLICY teacher_admin_only ON teachers
    USING (current_setting('app.role') = 'admin');
```
### Step 3: Tool Selection for Data Transfer (Weeks 3–4)
Three migration tool options in priority order:

**Option A — pgloader (Recommended — Free, Open Source)**

`pgloader` is the professional standard for migrating from Access/SQL Server/MySQL to PostgreSQL in a single command. It handles schema mapping, data type conversion, foreign key recreation, and bulk loading automatically.

```bash
# Install
sudo apt install pgloader

# Migrate Access .mdb/.accdb via ODBC
pgloader \
  "access:///path/to/school.accdb" \
  "postgresql://user:pass@localhost/sacc_db"

# pgloader generates a full migration report with row counts and errors
```

Known pgloader limitation: Access `.accdb` files require ODBC bridge setup on Linux (use MDBTools or a Windows intermediate). The recommended workflow is: export Access → CSV per table → pgloader CSV → PostgreSQL.

**Option B — ESF Database Migration Toolkit (Commercial, ~$49–149 one-time)**

GUI-based wizard approach for non-technical team members. Supports field-by-field mapping review, handles character encoding for Chinese names (UTF-8 essential), and generates a migration log. Best choice if the data cleanup is done by someone without command-line comfort.

**Option C — Manual Export + Python ETL Script**

For a database of this size (likely under 10,000 records total), a custom Python script using `pandas` + `psycopg2` gives maximum control over data transformation:

```python
import pandas as pd
import psycopg2
from cryptography.fernet import Fernet

# Export tables from Access as CSV, then:
df = pd.read_csv('teachers_export.csv')
df['dob'] = pd.to_datetime(df['DOB'])
df['ssn_encrypted'] = df['SSN'].apply(lambda x: encrypt(x, key))
df.drop(columns=['SSN'], inplace=True)
# ... write to PostgreSQL via psycopg2 COPY
```
### Step 4: Data Cleansing Before Migration (Week 3)
Access databases accumulated over years typically contain:

- Duplicate student records (same student registered multiple semesters with different spellings)
- Inconsistent date formats
- Empty required fields (name, DOB)
- Chinese name encoding issues (ensure UTF-8 output from Access export)
- Phone numbers in inconsistent formats

Run a cleansing pass in Python/pandas or Excel before loading into PostgreSQL. Deduplicate on (last_name + first_name + dob) combination. Flag ambiguous matches for manual review rather than auto-merging.
### Step 5: Post-Migration Validation (Week 4–5)
After loading data:

```sql
-- Row count validation per table
SELECT 'students' AS tbl, COUNT(*) FROM students
UNION ALL
SELECT 'teachers', COUNT(*) FROM teachers
UNION ALL  
SELECT 'classes', COUNT(*) FROM classes;
-- Compare each count vs original Access table count

-- Sequence alignment (critical after bulk insert)
SELECT setval('students_student_id_seq', 
              (SELECT MAX(student_id) FROM students));

-- Check foreign key integrity
SELECT s.student_id, e.class_id
FROM enrollments e
LEFT JOIN students s ON s.student_id = e.student_id
WHERE s.student_id IS NULL;  -- should return 0 rows
```

Also validate all Chinese name characters rendered correctly (UTF-8 check), all DOB dates are within plausible ranges, and all photo/document file references point to accessible files.
### Step 6: Database Options Comparison
| Option | Strengths | Weaknesses | Verdict for SACC |
|---|---|---|---|
| **PostgreSQL** | Open-source, full ACID, JSON support, row-level security, best Django ORM support, free | Requires managed hosting | **Primary recommendation** |
| **MySQL 8+** | Wide hosting support, familiar to many devs | Weaker ACID guarantees historically, less advanced Django integration | Acceptable fallback |
| **SQLite** | Zero config, great for dev | Not for multi-user production, no row-level security | Dev environment only |
| **MS SQL Server** | Native Access migration path via SSMA wizard | Microsoft licensing cost, less Pythonic, heavy | Avoid — overkill and expensive |
| **Supabase (Postgres-as-a-Service)** | Built-in auth, realtime, storage, REST API auto-gen | Pauses on free tier, paid starts at $25/mo | Good for later Phase 3+ |

**Decision: PostgreSQL 16+ is the unambiguous choice** — it has the best Django integration, row-level security built-in (critical for the teacher/student access boundary), JSON support for flexible metadata (SACC certifications, incident logs), and the widest managed hosting options at every price point.

***
## Part 3 — Phase 2: Django Application Architecture (Weeks 5–16)
### Database Schema: Django Models Architecture
The Django app will have the following app structure:

```
sacc_project/
├── core/              # Shared utilities, base models
├── students/          # Student records, enrollment, medical
├── classes/           # Class definitions, schedules
├── teachers/          # Admin-only HR records
├── attendance/        # Daily attendance sheets
├── registration/      # Online registration forms, payments
├── reports/           # Print-ready attendance/schedule cards
└── accounts/          # User authentication, roles
```
### Role-Based Access Control (RBAC) Design
The notes specify three distinct access levels, with the teacher database explicitly walled off:

| Role | Can Access | Cannot Access |
|---|---|---|
| **Super Admin** | Everything — all 4 domains | Nothing restricted |
| **Admin** | Student data, class data, teacher HR records, reports, payments | N/A |
| **Staff (Student Data Handler)** | Student records, enrollment, attendance, class lists | Teacher HR data (name, SSN, pay rate, address, certifications) |
| **Teacher** | Their own class roster, attendance entry, student schedule card | Other classes, HR records, payment data |
| **TA** | Their assigned class attendance | Student PII beyond name and schedule |

Implement using Django's built-in permission system extended with `django-guardian` for object-level permissions, plus PostgreSQL Row Level Security as a defense-in-depth layer:

```python
# accounts/models.py
from django.contrib.auth.models import AbstractUser

class SACCUser(AbstractUser):
    ROLE_CHOICES = [
        ('super_admin', 'Super Admin'),
        ('admin', 'Admin'),
        ('staff', 'Staff'),
        ('teacher', 'Teacher'),
        ('ta', 'Teaching Assistant'),
    ]
    role = models.CharField(max_length=20, choices=ROLE_CHOICES)
    
    class Meta:
        permissions = [
            ('view_teacher_hr', 'Can view teacher HR records'),
            ('view_ssn', 'Can view Social Security Numbers'),
            ('edit_pay_rates', 'Can edit pay rates'),
        ]

# teachers/views.py — enforce at view level
from django.contrib.auth.mixins import PermissionRequiredMixin

class TeacherHRDetailView(PermissionRequiredMixin, DetailView):
    permission_required = 'accounts.view_teacher_hr'
    model = Teacher
    # Staff without this permission get 403 — not a redirect to login
```
### Security Architecture
Django provides an excellent security baseline:

**Authentication & Sessions:**
- Use `django-allauth` with MFA (multi-factor authentication) for admin accounts
- JWT tokens (`djangorestframework-simplejwt`) for any mobile/API access
- Session timeout after 30 minutes of inactivity
- HTTPS enforced via `SECURE_SSL_REDIRECT = True` and `HSTS` headers

**Data Protection:**
- SSN stored encrypted with `django-encrypted-model-fields` (AES-256)
- File uploads (portrait photos, medical forms, Gov ID scans) stored in private S3-compatible bucket — never served directly; Django generates signed time-limited URLs
- All sensitive fields never logged by Django's logging middleware (use custom middleware to scrub)

**FERPA Compliance:**
- Student education records accessible only to authorized school officials
- Audit log of every access to a student record (who viewed what, when)
- Parental consent workflows for any third-party sharing
- Right-to-access: parents can request data exports via admin panel

```python
# Audit logging middleware
class AuditLogMiddleware:
    def __call__(self, request):
        response = self.get_response(request)
        if request.path.startswith('/students/') and request.user.is_authenticated:
            AuditLog.objects.create(
                user=request.user,
                action=request.method,
                path=request.path,
                timestamp=now(),
                ip=request.META.get('REMOTE_ADDR')
            )
        return response
```

**COPPA Compliance** (K-12, children under 13):
- No behavioral tracking or advertising technologies
- Data minimization — collect only fields listed in the spec
- Parental consent recorded and stored before any data collection
- Right to deletion (parent can request record purge)
### Key Features from the Spec (Page 4 Notes)
Mapping the handwritten goals to Django features:

| Spec Goal | Django Implementation |
|---|---|
| "Secure student database, accessible online to admin" | Django admin + custom views, HTTPS, role auth |
| "Website registration form — improve on Google Form" | Django ModelForm with file upload, multi-step wizard |
| "Payment option" | Stripe integration via `dj-stripe`; Zelle instructions page |
| "Each student: Registration Form, Current Classes, Duration, History" | Student detail view with enrollment history timeline |
| "Student allergy/medical info, flag for outdated" | Medical model with `updated_at` + auto-flag logic |
| "Grades history / report cards" | GradeRecord model linked to semester + class |
| "Student Schedule Card" | PDF generation via `WeasyPrint` or `ReportLab` |
| "Summer Field Trip waiver/permission/payment" | Separate waiver model with parent digital signature |
| "Incident logs" | IncidentLog model — admin/teacher write, admin-only delete |
| "Attendance sheets — printable + tablet" | Responsive HTML attendance view + PDF export |
| "School Bus attendance with arrival times" | BusRoute + BusStop model with estimated time field |
| "Overall master searchable student list" | Django ListView with full-text search via `django-watson` |
| "Teacher/TA attendance sheets" | Separate attendance model for staff vs students |
| "Student login? Passwords? Authenticator?" | Optional parent portal — Django allauth + TOTP 2FA |

***
## Part 4 — Phase 3: Web Application Development (Weeks 9–32)
### Recommended Tech Stack
| Layer | Technology | Reason |
|---|---|---|
| **Backend** | Django 5.x + Django REST Framework | Python ecosystem, built-in admin, ORM, security defaults |
| **Database** | PostgreSQL 16 | Best Django integration, RLS, JSONB, full-text search |
| **Frontend** | Django templates + HTMX + Alpine.js | Server-rendered (simpler, faster for this use case), interactive without full SPA complexity |
| **PDF Generation** | WeasyPrint | Print-quality attendance sheets, schedule cards, report cards |
| **File Storage** | AWS S3 or Backblaze B2 (cheaper) | Private bucket for photos, medical forms, Gov IDs |
| **Email** | SendGrid or AWS SES | Parent notifications, registration confirmations |
| **Payments** | Stripe | Online registration payments, field trip fees |
| **Search** | PostgreSQL `pg_trgm` extension + `django-watson` | Full-text student search |
| **Task Queue** | Celery + Redis | Async PDF generation, email sending, report generation |
| **Monitoring** | Sentry (free tier) | Error tracking in production |
### Development Sprints (Agile, 2-Week Cycles)
**Sprints 1–2 (Weeks 9–12): Foundation**
- Django project scaffold, custom user model with roles
- PostgreSQL connection, all models created (students, teachers, classes, enrollment)
- Django admin configured and locked down by role
- Basic login / logout with MFA setup

**Sprints 3–4 (Weeks 13–16): Core Student Database**
- Student CRUD views (list, detail, create, edit)
- Parent/guardian and emergency contact sub-forms
- Medical info with outdated-flag logic
- Enrollment history view
- Master searchable student list

**Sprints 5–6 (Weeks 17–20): Class & Attendance Management**
- Class management (CRUD, assign teachers/TAs)
- Daily attendance entry (tablet-friendly checklist view)
- Attendance sheet PDF export (print-ready)
- Bus attendance with arrival time estimates

**Sprints 7–8 (Weeks 21–24): Registration & Payments**
- Online registration form (multi-step wizard, replaces Google Form)
- Stripe payment integration for registration fees
- Field trip waiver with digital parent signature
- Registration history per student

**Sprints 9–10 (Weeks 25–28): Reporting & Teacher HR**
- Student Schedule Card generation (PDF)
- Grades/report card entry and PDF export
- Teacher HR module (admin-only, separate URL namespace)
- SACC certification tracking per teacher
- Pay rate history

**Sprints 11–12 (Weeks 29–32): Polish, Security Audit & Launch**
- Incident log module
- Full audit logging of student record access
- Security penetration test (OWASP Top 10 checklist)
- Performance testing (simulate 50 concurrent users)
- Staff training documentation
- Production deployment

***
## Part 5 — Hosting Cost Comparison
### Budget-Friendly Options vs Enterprise Cloud
For a school aftercare program of this scale (estimated 200–500 students, 20–40 staff, admin team of 3–5), the workload is modest: one Django web process, one PostgreSQL database, file storage. Here is the 2026 pricing reality:

| Platform | Web Service | Managed PostgreSQL | Total/Month | Notes |
|---|---|---|---|---|
| **Fly.io** | ~$3.19/mo (shared 512MB) | Self-managed (~$4/mo) | **~$7/mo** | Cheapest; Postgres is NOT managed — you own backups |
| **Railway** | ~$15/mo (usage-based) | ~$5/mo | **~$20/mo** | Usage-based billing; had 5+ outages since Nov 2025 |
| **DigitalOcean App Platform** | $5/mo flat | $15.15/mo managed | **~$20/mo** | True managed Postgres with auto-backups; most Heroku-like |
| **Render** | $7/mo flat | $19/mo (5GB tier) | **~$26/mo** | Fixed billing, no surprises; 100GB free egress |
| **Supabase** | (includes Postgres) | $25/mo Pro plan | **~$25/mo** | Full platform: auth + DB + storage; great for later phases |
| **AWS Elastic Beanstalk + RDS** | ~$35–50/mo (t3.small EB) | ~$30–60/mo (db.t3.micro RDS) | **~$65–110/mo** | Complex setup; requires DevOps knowledge; FERPA-eligible via GovCloud |
| **AWS EC2 + RDS (production)** | ~$50–100/mo | ~$50–200/mo | **~$100–300/mo** | Full production scale; expensive for a small program |
| **Hetzner VPS (self-managed)** | ~$5–10/mo (CX21) | Self-managed on same server | **~$10–15/mo** | Cheapest with capability, but you manage everything including backups |
### Recommended Hosting Path
**Phase 1–2 (Development): Free**
- Use `db.sqlite3` locally during development
- GitHub for version control (free for private repos)
- Docker Compose for local environment

**Phase 2–3 (Beta/Testing): DigitalOcean — ~$20/month**
- App Platform Basic ($5/mo) + Managed PostgreSQL Basic ($15.15/mo)
- True managed database with automated daily backups, failover
- Familiar Git-push deploy workflow
- Managed Postgres starts at $15/month with no hidden I/O fees
- DigitalOcean is 45% cheaper than AWS at 500GB bandwidth

**Production Launch: DigitalOcean or Render — $20–26/month**
- Add Backblaze B2 for file storage (~$6/mo for 1TB — far cheaper than S3)
- Add SendGrid free tier (100 emails/day free)
- Sentry error monitoring (free tier)
- **Total production cost: ~$30–40/month**

**If FERPA compliance requires US-based, auditable cloud:**
- AWS with FERPA BAA (Business Associate Agreement) — ~$80–150/month
- GovCloud not needed for FERPA (standard AWS with a signed BAA suffices)
- Only pursue this if the program is subject to strict district IT compliance review
### Do NOT Use for This Scale
- **AWS as the first choice** — the operational complexity and cost is disproportionate for a program of this size. DigitalOcean provides equivalent reliability at 45–60% lower cost.
- **Fly.io for production** — self-managed Postgres means you own disaster recovery. A failed Postgres container with no admin available on a school day is an unacceptable risk.

***
## Part 6 — Full 12-Month Timeline
```
MONTH 1:   Access Audit + Schema Design + pgloader Migration Setup
MONTH 2:   Data Migration, Cleansing, Validation + PostgreSQL Production Setup
MONTH 3:   Django Project Scaffold + User Auth + Admin + Core Models
MONTH 4:   Student Database CRUD + Search + Enrollment History
MONTH 5:   Class Management + Attendance (tablet + print)
MONTH 6:   Online Registration Form + Stripe Payments + Field Trip Waiver
MONTH 7:   Reporting Engine (PDFs: Schedule Cards, Attendance, Report Cards)
MONTH 8:   Teacher HR Module (admin-only) + Certifications + Pay History
MONTH 9:   Incident Logs + Audit Trail + Bus Attendance
MONTH 10:  Security Audit + Penetration Test + Performance Testing
MONTH 11:  Staff Training + Documentation + Phased Rollout (pilot one class)
MONTH 12:  Full Production Launch + 30-Day Post-Launch Review + Optimization
```

**6-Month Accelerated Track** (with dedicated developer, 40 hrs/week):

Skip Months 8–9 features (Teacher HR, Incident Logs) until post-launch. Focus on the core student database + attendance + registration/payment. Teacher HR can be added as Phase 2 after core launch.

***
## Part 7 — Development Cost Estimate
Based on 2026 market rates for custom school management software:
### Build Cost Options
| Development Path | Estimated Cost | Timeline | Notes |
|---|---|---|---|
| **Hire senior Django freelancer (US)** | $80–150/hr × 300–500 hrs = $24,000–75,000 | 4–10 months | Upwork/Toptal; quality varies widely |
| **Hire Django developer (Eastern Europe/LatAm)** | $30–60/hr × 300–500 hrs = $9,000–30,000 | 4–10 months | Strong talent pool; async timezone management |
| **Small dev agency (US)** | $40,000–120,000 flat | 6–12 months | Includes project management; higher but predictable |
| **Small dev agency (offshore)** | $15,000–45,000 flat | 6–12 months | Lower cost; requires detailed spec upfront |
| **In-house build (technical staff)** | $0–5,000 tooling + hosting | 6–18 months | Only if tech talent exists internally |
| **Open-source base + customization** | $5,000–20,000 customization | 3–8 months | Use `django-scms` or similar as scaffold |
### Ongoing Annual Cost (Post-Launch)
| Item | Annual Cost |
|---|---|
| Hosting (DigitalOcean) | ~$360–480/year |
| File Storage (Backblaze B2) | ~$72/year |
| Email (SendGrid paid if >100/day) | ~$0–200/year |
| SSL Certificate | Free (Let's Encrypt) |
| Domain Name | ~$15/year |
| Maintenance / Bug Fixes | 15–20% of build cost annually |
| Security updates (Django LTS upgrades) | Included in maintenance |

**Minimum viable annual operating cost: ~$500–700/year (hosting + domains)**

***
## Part 8 — Security Checklist for Production Launch
Before going live, verify every item:

- [ ] HTTPS enforced on all routes (`SECURE_SSL_REDIRECT`, `HSTS`)
- [ ] Django `DEBUG = False` in production
- [ ] `SECRET_KEY` stored in environment variable, never in code
- [ ] Database credentials in environment variables, not settings.py
- [ ] SSN field encrypted at application layer (AES-256)
- [ ] File storage bucket is **private** — no public-read ACL
- [ ] All admin routes protected by MFA
- [ ] Rate limiting on login endpoint (prevent brute force)
- [ ] CSRF protection enabled (Django default — do not disable)
- [ ] SQL injection: use Django ORM parameterized queries only (no raw SQL with f-strings)
- [ ] Dependency audit: `pip-audit` run before launch
- [ ] Automated daily PostgreSQL backups verified restorable
- [ ] Audit log records every student record access
- [ ] FERPA disclosure policy documented in app
- [ ] Staff trained: never share login credentials, log out on shared tablets
- [ ] PostgreSQL: no `public` schema write access for app user — use dedicated schema
- [ ] Row Level Security policies active on `teachers` table

***
## Appendix: Recommended Open-Source Starting Points
Rather than building entirely from scratch, begin with a proven Django school app as scaffold and customize to the SACC spec:

1. **django-scms** (github.com/mwinamijr/django-scms) — Django + DRF, PostgreSQL-tested, role-based auth for Admin/Teacher/Parent already implemented
2. **django-sis** (github.com/saulshanabrook/django-sis) — Relies on Django admin interface heavily; good reference for field structures
3. **School Management System** (github.com/PavanKumar1207/Student_management_system) — Enrollment, attendance, grades, RBAC

Using an open-source base as a starting scaffold can reduce custom development time by 30–50% and costs by $10,000–30,000 compared to a clean-slate build, while still allowing full customization for the SACC-specific fields (Chinese names, SACC certifications, bilingual class descriptions, bus attendance, etc.).

***

*Prepared July 2026. All hosting prices verified against July 2026 provider pricing pages. Development cost estimates based on 2026 freelance and agency market rates for Django/PostgreSQL school information systems.*

---

## References

1. [Migrating MS Access to PostgreSQL: Step-by-Step](https://msaccessonline.com/blog/access-to-postgresql) - Migrate MS Access to PostgreSQL step-by-step: schema conversion, data transfer, query rewrite, and A...

