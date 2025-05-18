-- BookDB PostgreSQL initialization script
-- This script runs when the PostgreSQL container is first initialized
-- Only installs necessary extensions but doesn't prepopulate data

-- Create extensions if needed
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm"; -- For text search
