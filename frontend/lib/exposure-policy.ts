export const profileExposurePolicy = {
  hideCountry: true,
  hideMilitaryStatus: true,
  lowerPriorityBachelorsField: true,
};

export function lowPriorityLabel(value: string | null | undefined): string {
  if (!value || value === "해당없음") return "전공 정보 낮음";
  return `전공 ${value}`;
}
