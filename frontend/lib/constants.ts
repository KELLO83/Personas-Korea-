export const DEFAULT_PERSONA_UUID = "a5ad493e75e74e5cb4a81ac934a1db8f";

export const NODE_TYPE_LABELS: Record<string, string> = {
  Person: "페르소나",
  Hobby: "취미",
  Skill: "스킬",
  Occupation: "직업",
  District: "지역",
  Province: "시도",
  EducationLevel: "학력",
  HousingType: "주거",
  FamilyType: "가구",
  MaritalStatus: "혼인",
  MilitaryStatus: "병역",
  Country: "국가",
  Field: "전공",
};

export const RELATION_LABELS: Record<string, string> = {
  ENJOYS_HOBBY: "취미",
  LIKES: "취미",
  HAS_SKILL: "보유 스킬",
  WORKS_AS: "직업",
  LIVES_IN: "거주 지역",
  LIVES_IN_HOUSING: "주거 형태",
  LIVES_WITH: "가구 형태",
  EDUCATED_AT: "학력",
  MARITAL_STATUS: "혼인 상태",
  MILITARY_STATUS: "병역 상태",
  SIMILAR_TO: "유사 페르소나",
};

export const NODE_COLORS: Record<string, string> = {
  Person: "#74d6ff",
  Hobby: "#8bff8b",
  Skill: "#ffd166",
  Occupation: "#d7a8ff",
  District: "#ff7b7b",
  Province: "#f88cff",
  EducationLevel: "#66f0d6",
  HousingType: "#b8e986",
  FamilyType: "#f8e71c",
  MaritalStatus: "#ffb6d5",
  MilitaryStatus: "#a8aeb8",
};

export const NODE_EMOJIS: Record<string, string> = {
  Person: "👤",
  Hobby: "🎮",
  Skill: "🛠️",
  Occupation: "💼",
  District: "📍",
  Province: "🗺️",
  EducationLevel: "🎓",
  HousingType: "🏠",
  FamilyType: "👨‍👩‍👧",
  MaritalStatus: "💍",
  MilitaryStatus: "🪖",
};
