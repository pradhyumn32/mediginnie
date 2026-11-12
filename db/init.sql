CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS btree_gin;

CREATE TABLE IF NOT EXISTS drug_interactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    drug_a VARCHAR NOT NULL,
    drug_b VARCHAR NOT NULL,
    severity VARCHAR NOT NULL DEFAULT 'MAJOR',
    description TEXT NOT NULL,
    clinical_management TEXT,
    source VARCHAR DEFAULT 'clinical_pairwise_catalog',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS ix_drug_interactions_pair ON drug_interactions (drug_a, drug_b);

-- Initial high-risk clinically validated pairwise interactions
INSERT INTO drug_interactions (drug_a, drug_b, severity, description, clinical_management)
VALUES
    ('sildenafil', 'nitroglycerin', 'CONTRAINDICATED', 'Co-administration of PDE5 inhibitors (sildenafil) with organic nitrates causes profound, potentially fatal systemic vasodilation and hypotension.', 'Avoid concomitant use. Nitrates are strictly contraindicated within 24 hours of sildenafil administration.'),
    ('warfarin', 'aspirin', 'MAJOR', 'Concurrent use increases risk of severe upper gastrointestinal and systemic hemorrhage due to combined antiplatelet and anticoagulant effects.', 'Monitor INR closely. Consider gastroprotective therapy (e.g. PPI) if co-therapy is clinically essential.'),
    ('simvastatin', 'clarithromycin', 'CONTRAINDICATED', 'Clarithromycin is a potent CYP3A4 inhibitor that dramatically increases simvastatin serum concentrations, precipitating severe rhabdomyolysis and acute renal failure.', 'Suspend simvastatin during macrolide antibiotic therapy or substitute with an agent not metabolized by CYP3A4 (e.g. pravastatin).'),
    ('lisinopril', 'spironolactone', 'MAJOR', 'Combination of an ACE inhibitor with a potassium-sparing aldosterone antagonist creates a severe risk of life-threatening hyperkalemia and cardiac arrhythmias.', 'Monitor serum potassium and renal function within 1 week of initiation and regularly thereafter.'),
    ('fluoxetine', 'phenelzine', 'CONTRAINDICATED', 'Co-administration of SSRIs with monoamine oxidase inhibitors (MAOIs) can precipitate fatal serotonin syndrome with hyperthermia, rigidity, and autonomic instability.', 'A 5-week washout period is mandatory after discontinuing fluoxetine before initiating an MAOI.');

