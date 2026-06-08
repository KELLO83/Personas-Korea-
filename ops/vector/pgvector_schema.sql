CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS {table_name} (
    id BIGSERIAL PRIMARY KEY,
    person_uuid UUID NOT NULL UNIQUE,
    display_name TEXT,
    age INTEGER,
    age_group TEXT,
    sex TEXT,
    province TEXT,
    district TEXT,
    occupation TEXT,
    marital_status TEXT,
    military_status TEXT,
    family_type TEXT,
    housing_type TEXT,
    education_level TEXT,
    bachelors_field TEXT,
    skills TEXT[],
    hobbies TEXT[],
    persona_text TEXT NOT NULL,
    embedding_text TEXT,
    embedding vector({dimension}),
    source_model TEXT,
    embedding_dim SMALLINT,
    embedding_text_version TEXT NOT NULL DEFAULT 'persona_embedding_v1',
    metadata JSONB NOT NULL DEFAULT '{{}}'::JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS {table_name}_age_group_idx
ON {table_name} (age_group);

CREATE INDEX IF NOT EXISTS {table_name}_sex_idx
ON {table_name} (sex);

CREATE INDEX IF NOT EXISTS {table_name}_province_idx
ON {table_name} (province);

CREATE INDEX IF NOT EXISTS {table_name}_district_idx
ON {table_name} (district);

CREATE INDEX IF NOT EXISTS {table_name}_occupation_idx
ON {table_name} (occupation);

CREATE INDEX IF NOT EXISTS {table_name}_embedding_text_version_idx
ON {table_name} (embedding_text_version);

CREATE INDEX IF NOT EXISTS {table_name}_skills_gin_idx
ON {table_name}
USING gin (skills);

CREATE INDEX IF NOT EXISTS {table_name}_hobbies_gin_idx
ON {table_name}
USING gin (hobbies);

CREATE INDEX IF NOT EXISTS {table_name}_hnsw_idx
ON {table_name}
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
