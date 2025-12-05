# Database Backup & Restore Drill Runbook

> **Doc Status (2025-11-30): Active Runbook**
> **Role:** Step-by-step practice drill for verifying that Postgres backups and restores actually work in staging or a non-production environment.
>
> **SSoT alignment:** This runbook is a derived operational procedure over:
>
> - `docs/OPERATIONS_DB.md` – canonical database operations and migration workflows
> - `docs/DATA_LIFECYCLE_AND_PRIVACY.md` – data retention, minimisation, and recovery expectations
> - `docs/DEPLOYMENT_REQUIREMENTS.md` / `docker-compose*.yml` – deployment topology and volumes
> - `docs/SECRETS_MANAGEMENT.md` – secrets inventory, DB credential handling, and rotation patterns
> - `docs/runbooks/SECRETS_ROTATION_DRILL.md` – complementary PASS-style drill for JWT and DB credential rotation
>
> **Precedence:** Postgres configuration, Prisma migrations, and deployment manifests are authoritative for actual behaviour. If this runbook disagrees with them, **code + configs win** and this document must be updated.

---

## 1. Goal & Scope

This runbook defines a **non-destructive backup and restore drill** for the RingRift Postgres database.

The goal is to regularly prove that you can:

- Take a usable logical backup (`pg_dump`) of the main database.
- Restore that backup into a **separate database** on the same Postgres instance (or a throwaway instance).
- Run a small application-level smoke test against the restored database.

This should be performed in **staging** or an equivalent non-production environment. Do **not** run this drill directly against production without adapting it to your provider’s backup/restore mechanisms.

---

## 2. Preconditions & Safety

- You are operating in **staging** (or another non-production environment).
- The Postgres container is named `postgres` and mounts `./backups` → `/backups`
  (see `docker-compose.yml` and `docs/OPERATIONS_DB.md` §1.1–1.2).
- The primary database is named `ringrift` and owned by user `ringrift`.
- You have shell access to the host where Docker Compose is running.
- The staging stack is on a known-good commit and the database schema matches the
  current migrations on `main` (that is, `prisma/schema.prisma` + `prisma/migrations`
  have been rolled out via `npx prisma migrate deploy` as described in
  `docs/OPERATIONS_DB.md` §2).
- The **monitoring stack is active** for the environment (Prometheus, Alertmanager,
  Grafana per `docs/DEPLOYMENT_REQUIREMENTS.md`) so you can watch database/HTTP
  health during the drill.
- There is sufficient free space in the backup target (for the default Compose
  setup, the `./backups` directory on the host).

> **Safety principle:** This drill restores into a **new database name** (for example `ringrift_restore_drill`) and never drops or overwrites the primary `ringrift` database.

If your topology or naming differs, adapt the commands accordingly but keep the “separate restore DB” principle.

Before you start, it is recommended (from the project root) to run:

```bash
npm run validate:deployment           # validate docker-compose, env schema, CI wiring
npm run validate:monitoring           # validate Prometheus / Alertmanager configs (optional but recommended)
npm run ssot-check                    # docs + config SSoT checks, including data-lifecycle and secrets docs (optional but recommended)
```

These give you a clean baseline so any issues uncovered during the drill can be attributed to the backup/restore process rather than pre-existing configuration drift.

### 2.1 PASS framing (staging drill overview)

- **Purpose:** Prove that you can restore a recent logical backup of the primary
  staging database into a **separate** staging-like database and exercise
  core application flows against it.
- **Preconditions:** As listed above: non-production environment, healthy
  `postgres` service, migrations in sync with `main`, monitoring online,
  and enough space for a fresh `pg_dump`.
- **Actions (high level):**
  1. Take a fresh logical backup of the staging database into the configured
     backup location (default `./backups`).
  2. Restore that backup into a new database (for example `ringrift_restore_drill`)
     on the same Postgres instance or a throwaway instance.
  3. Validate the restored database:
     - Schema and migrations (`prisma migrate status`).
     - Optional app-level health and auth/game flows using the existing
       health endpoints and smoke tools.
  4. Clean up the restore database and any temporary containers.
- **Signals & KPIs:**
  - `psql '\dt'` and targeted queries against the restore DB show the expected
    schema and data.
  - `prisma migrate status` reports that all expected migrations are applied
    to the restore DB.
  - If you temporarily point an app instance at the restore DB, `/health` and
    `/ready` stay green and core auth/game flows succeed without integrity
    or foreign-key errors.
  - No unexpected spikes in database connectivity errors or latency in
    Grafana dashboards during the drill.

---

## 3. Step-by-step Drill (staging with docker-compose)

All commands below assume you are on the staging host in the RingRift deployment directory (for example `/opt/ringrift`).

### 3.1 Take a fresh logical backup

1. Confirm the `postgres` container is healthy:

   ```bash
   docker compose ps postgres
   ```

2. Take a timestamped logical backup of the `ringrift` database:

   ```bash
   TIMESTAMP=$(date +%Y%m%d_%H%M%S)
   docker compose exec postgres \
     pg_dump -U ringrift -d ringrift \
     -f /backups/staging_drill_${TIMESTAMP}.sql
   ```

3. Verify the backup file exists on the host:

   ```bash
   ls -lh backups/staging_drill_*.sql | tail -1
   ```

4. (Optional) Spot-check the file header:

   ```bash
   head -20 backups/staging_drill_${TIMESTAMP}.sql
   ```

---

### 3.2 Create a separate restore database

Create a new database for the drill restore so that the primary `ringrift` database is untouched:

```bash
RESTORE_DB=ringrift_restore_drill

docker compose exec postgres \
  createdb -U ringrift "${RESTORE_DB}"

# Confirm it exists
docker compose exec postgres \
  psql -U ringrift -lqt | grep "${RESTORE_DB}" || true
```

If `createdb` fails because the database already exists, either drop it explicitly (only if you are certain it is safe), or choose a different name such as `ringrift_restore_drill_YYYYMMDD`.

---

### 3.3 Restore from the backup into the drill database

1. Restore the most recent drill backup into the new database:

   ```bash
   RESTORE_DB=ringrift_restore_drill
   LATEST_BACKUP=$(ls -1 backups/staging_drill_*.sql | sort | tail -1)

   echo "Restoring from ${LATEST_BACKUP} into ${RESTORE_DB}…"

   docker compose exec -T postgres \
     psql -U ringrift -d "${RESTORE_DB}" < "${LATEST_BACKUP}"
   ```

2. Verify core tables exist:

   ```bash
   docker compose exec postgres \
     psql -U ringrift -d "${RESTORE_DB}" -c '\dt'
   ```

   You should see the same schema tables you expect from `ringrift`.

---

### 3.4 Validation and smokes against the restored DB

The goal of this step is to prove that the restored database is not only
structurally valid, but also **usable by the application** and aligned with
current migrations.

1. Construct a temporary `DATABASE_URL` pointing at the restore DB. For a
   Compose-based Postgres, this often looks like:

   ```bash
   export DATABASE_URL="postgresql://ringrift:${DB_PASSWORD:-password}@postgres:5432/${RESTORE_DB}"
   echo "${DATABASE_URL}"
   ```

   Adapt user, password, and host to match your staging configuration and
   `ENVIRONMENT_VARIABLES.md`.

2. Run a Prisma status check against the restored database from a one-off
   `app` container:

   ```bash
   docker compose run --rm \
     -e NODE_ENV=production \
     -e DATABASE_URL="${DATABASE_URL}" \
     app npx prisma migrate status
   ```

   This should report that all expected migrations are applied.

3. (Optional but recommended) Run a light application smoke against the
   restored DB:
   - Start a one-off app container pointed at the restore DB:

     ```bash
     docker compose run --rm \
       -e NODE_ENV=production \
       -e DATABASE_URL="${DATABASE_URL}" \
       -p 4000:3000 \
       app npm run start
     ```

   - In another shell, hit health endpoints for the temporary app:

     ```bash
     curl -s http://localhost:4000/health | jq
     curl -s http://localhost:4000/ready | jq
     ```

   - Optionally create a throwaway user and a short game via the usual
     UI/API flows to ensure reads and writes succeed.

4. (Optional but recommended) Reuse existing validation tooling while the
   temporary app is pointing at the restored DB (or after temporarily
   pointing your normal staging stack at `${RESTORE_DB}`):
   - From the project root, re-run config and docs validations:

     ```bash
     npm run validate:deployment
     npm run validate:monitoring      # recommended
     npm run ssot-check               # recommended
     ```

   - Run an auth smoke against the app that is using the restored DB
     (either the temporary app on port 4000 with adjusted URLs, or the
     main staging app if you have pointed it at `${RESTORE_DB}`):

     ```bash
     # Example when the app is reachable at http://localhost:3000
     ./scripts/test-auth.sh
     ```

   - Optionally run a small orchestrator/game load smoke to exercise the
     restored DB under light load:

     ```bash
     # App reachable at the default http://localhost:3000
     npm run load:orchestrator:smoke
     ```

Stop the temporary app container (or revert your staging stack to point back
at the primary `ringrift` database) once you are satisfied with the results.

---

### 3.5 Cleanup

After the drill is complete and validated:

1. Drop the restore database (if no longer needed):

   ```bash
   docker compose exec postgres \
     dropdb -U ringrift ringrift_restore_drill
   ```

   Adjust the name if you used a timestamped variant.

2. Optionally prune old drill backups, keeping at least the most recent successful one:

   ```bash
   ls -1 backups/staging_drill_*.sql
   # Remove only files you are sure are safe to delete
   # rm backups/staging_drill_YYYYMMDD_HHMMSS.sql
   ```

Never delete production backups as part of this drill.

---

## 4. Validation Checklist

- [ ] `postgres` container healthy and reachable.
- [ ] New logical backup file created under `./backups/` for the drill run.
- [ ] Separate restore database (for example `ringrift_restore_drill`) created successfully.
- [ ] `psql '\dt'` against the restore DB shows expected tables.
- [ ] `prisma migrate status` passes against the restore DB.
- [ ] Optional: temporary app instance can start against the restore DB and pass basic health checks (`/health`, `/ready`).
- [ ] Optional: `npm run validate:deployment` has been re-run after the drill to confirm deployment/config SSoTs are still consistent.
- [ ] Optional: `npm run validate:monitoring` and `npm run ssot-check` have been run and are green (no new warnings or drift introduced by the drill).
- [ ] Optional: `./scripts/test-auth.sh` and, if appropriate, `npm run load:orchestrator:smoke` succeed against an app instance pointed at the restored DB (or after the staging stack has been restored to normal configuration).
- [ ] Restore database dropped (or clearly labelled and left for further analysis).

Record the date, environment, and any issues found in your internal incident / ops log so you have a history of completed drills. If this drill is being run together with the secrets-rotation drill, cross-reference both runbooks in your log entry.

---

## 5. Adapting to Managed Postgres Providers

If staging or production uses a managed Postgres service (for example AWS RDS, GCP Cloud SQL, Azure Database for PostgreSQL):

- Replace the `docker compose exec postgres …` commands with provider-native mechanisms:
  - Snapshot creation / PITR configuration from the cloud console or CLI.
  - Restoring a snapshot into a **new instance** and pointing an app instance at it (or at a read-replica), rather than performing an in-place destructive restore on the primary.
- Still perform an **application-level smoke** against the restored instance:
  - Run `npx prisma migrate status` against the restored `DATABASE_URL`.
  - Run the same login / game creation / short-play flow used in `docs/OPERATIONS_DB.md` §2.4, and, where appropriate, the auth/load smokes referenced above.

For **production** restore drills, additionally:

- Align with the backup and privacy constraints in `docs/DATA_LIFECYCLE_AND_PRIVACY.md` §3.5 (“Backups and offline copies”) – especially the expectation that restores are followed by re-application of deletion/anonymisation routines where appropriate.
- Ensure DB credentials and `DATABASE_URL` values involved in the drill follow the rotation and access-control guidance in `docs/SECRETS_MANAGEMENT.md` and the `SECRETS_ROTATION_DRILL` runbook.
- Prefer blue/green or read-replica based techniques so that restores can be validated in isolation before any traffic is cut over.

The core principles remain the same:

- Backups are only useful if you can restore them.
- Restores should be proven **before** an incident, not during one.
- For drills in non-production, restored databases and instances should be treated as disposable once validation is complete.

---

## 6. Rollback & Safety Notes

### 6.1 Staging / non-production

- This drill never drops or overwrites the primary staging database `ringrift`.
- If any step fails:
  - Stop using the restore DB (shut down any app instance pointing at it).
  - Keep the primary `ringrift` DB untouched; normal staging traffic continues to use it.
  - Use logs and monitoring to diagnose the failure; you can safely drop and recreate the restore database and re-run the drill later.
- Restored databases created solely for this drill should be treated as **disposable** once validation is complete (drop them or clearly label them as ephemeral).

### 6.2 Production adaptation (high-level)

- Do not run this exact sequence directly against production. Instead:
  - Plan a dedicated production restore drill using a new instance or read-replica restored from a production backup.
  - Coordinate with incident management and change-control processes.
  - Follow the higher-level disaster-recovery flows in `docs/OPERATIONS_DB.md` §3 (Rollback & Disaster-Recovery Playbooks) and `docs/DATA_LIFECYCLE_AND_PRIVACY.md` §3.5 for privacy-aware handling of restored data.
- Any in-place destructive restore of a production database (dropping or overwriting the primary) requires a separate, more conservative runbook and is out of scope here.

---

## 7. Integration with Secrets & Other Runbooks

This drill is intentionally complementary to the secrets rotation drill:

- **Secrets & env configuration**
  - For the database credentials and `DATABASE_URL` values used during backup and restore, follow `docs/SECRETS_MANAGEMENT.md` and the **Database Password Rotation** section of the secrets rotation drill (`docs/runbooks/SECRETS_ROTATION_DRILL.md`).
  - Never hard-code live credentials into backup or restore commands; prefer environment variables and secret managers.
- **Combined drills**
  - For a more advanced exercise, you can chain this runbook with `SECRETS_ROTATION_DRILL` in staging:
    1. Run this backup/restore drill to prove you can restore a recent staging backup into a new database.
    2. Rotate the DB user/password following the secrets rotation drill while still pointing a staging or temporary app at the restored database.
    3. Validate that both the restored data and the new credentials behave correctly under auth/game smokes and monitoring.
  - When you record the drill in your internal ops log, note whether it was:
    - “Backup/restore only”, or
    - “Backup/restore + secrets rotation (combined drill)”.

This runbook remains **non-SSoT operational guidance** and defers to:

- `docs/DATA_LIFECYCLE_AND_PRIVACY.md` for data retention, anonymisation, and backup/privacy constraints.
- `docs/ENVIRONMENT_VARIABLES.md` for authoritative environment variable definitions such as `DATABASE_URL`.
- `docs/OPERATIONS_DB.md` for environment-specific DB topology and migration workflows.
- `docs/SECRETS_MANAGEMENT.md` and `docs/runbooks/SECRETS_ROTATION_DRILL.md` for secrets handling and rotation patterns.
