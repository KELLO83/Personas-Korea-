PERSON_LABEL = "Person"

ENTITY_LABELS = {
    "country": "Country",
    "province": "Province",
    "district": "District",
    "occupation": "Occupation",
    "skill": "Skill",
    "hobby": "Hobby",
    "education_level": "EducationLevel",
    "field": "Field",
    "marital_status": "MaritalStatus",
    "military_status": "MilitaryStatus",
    "family_type": "FamilyType",
    "housing_type": "HousingType",
}

PERSON_PROPERTY_FIELDS = [
    "uuid",
    "display_name",
    "age",
    "age_group",
    "sex",
    "persona",
    "professional_persona",
    "sports_persona",
    "arts_persona",
    "travel_persona",
    "culinary_persona",
    "family_persona",
    "cultural_background",
    "bachelors_field",
    "skills_and_expertise",
    "hobbies_and_interests",
    "career_goals_and_ambitions",
    "embedding_text",
]

CONSTRAINT_QUERIES = [
    "CREATE CONSTRAINT person_uuid_unique IF NOT EXISTS FOR (p:Person) REQUIRE p.uuid IS UNIQUE",
    "CREATE CONSTRAINT country_name_unique IF NOT EXISTS FOR (n:Country) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT province_name_unique IF NOT EXISTS FOR (n:Province) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT district_key_unique IF NOT EXISTS FOR (n:District) REQUIRE n.key IS UNIQUE",
    "CREATE CONSTRAINT occupation_name_unique IF NOT EXISTS FOR (n:Occupation) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT skill_name_unique IF NOT EXISTS FOR (n:Skill) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT hobby_name_unique IF NOT EXISTS FOR (n:Hobby) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT education_level_name_unique IF NOT EXISTS FOR (n:EducationLevel) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT field_name_unique IF NOT EXISTS FOR (n:Field) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT marital_status_name_unique IF NOT EXISTS FOR (n:MaritalStatus) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT military_status_name_unique IF NOT EXISTS FOR (n:MilitaryStatus) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT family_type_name_unique IF NOT EXISTS FOR (n:FamilyType) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT housing_type_name_unique IF NOT EXISTS FOR (n:HousingType) REQUIRE n.name IS UNIQUE",
]

INDEX_QUERIES = [
    "CREATE INDEX person_age IF NOT EXISTS FOR (p:Person) ON (p.age)",
    "CREATE INDEX person_age_group IF NOT EXISTS FOR (p:Person) ON (p.age_group)",
    "CREATE INDEX person_sex IF NOT EXISTS FOR (p:Person) ON (p.sex)",
    "CREATE INDEX person_display_name IF NOT EXISTS FOR (p:Person) ON (p.display_name)",
]


def schema_queries() -> list[str]:
    return CONSTRAINT_QUERIES + INDEX_QUERIES
