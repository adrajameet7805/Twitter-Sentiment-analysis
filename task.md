# Refactor & Security Improvements – Execution Checklist

## Phase 1 – Config / Environment
- [ ] Create .env
- [ ] Create .env.example
- [ ] Update .gitignore
- [ ] Update requirements.txt

## Phase 2 – Backend Fixes
- [ ] backend/auth_utils.py – dotenv for JWT secret
- [ ] backend/database.py – add username col, seed users, get_user_by_username
- [ ] backend/video_processor.py – remove auto-pip install
- [ ] models/inference_engine.py – remove dead code lines 168-171

## Phase 3 – App Entry Point
- [ ] app.py – remove JSON auth, add bcrypt validation, session guard

## Phase 4 – Frontend JS Update
- [ ] frontend/admin_login.py – update handleSignIn redirect to include credentials

## Phase 5 – File Deletions
- [ ] Delete root duplicate train scripts (6 files)
- [ ] Delete unused root inference scripts (4 files)
- [ ] Delete legacy model pkl files (3 files in models/)
