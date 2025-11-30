import pandas as pd
import random
import string

# -----------------------------------------------------
# CONFIG
# -----------------------------------------------------

LOCATIONS = [
    "Hà Nội", "TP HCM", "Đà Nẵng", "Hải Phòng", "Cần Thơ",
    "Huế", "Nha Trang", "Vũng Tàu", "Buôn Ma Thuột"
]

FAKE_PEOPLE = [
    "Nguyễn Văn T.", "Trần Minh K.", "Lê Thị H.",
    "Phạm Quốc D.", "Võ Hoàng P.", "Đỗ Khánh N."
]

COUNTERFACTUAL_STARTERS = [
    "Trái ngược với thông tin ban đầu, ",
    "Bất ngờ hơn so với báo cáo gốc, ",
    "Nguồn tin địa phương cho biết điều hoàn toàn khác: ",
]

EXAGGERATION_PHRASES = [
    "gây chấn động dư luận",
    "dẫn đến sự hỗn loạn chưa từng có",
    "ảnh hưởng nghiêm trọng tới toàn bộ khu vực",
    "làm cho người dân hoang mang tột độ",
]

# -----------------------------------------------------
# HELPERS
# -----------------------------------------------------

def fake_url():
    letters = ''.join(random.choice(string.ascii_lowercase) for _ in range(8))
    num = random.randint(1000, 9999)
    return f"https://fake-news.{letters}/{num}.html"

def distort_number(text: str):
    """Replace numbers in content with exaggerated ones."""
    if not isinstance(text, str):
        return text

    out = ""
    temp = ""
    for c in text:
        if c.isdigit():
            temp += c
        else:
            if temp != "":
                # Generate a random number unrelated to the original
                new_num = str(random.randint(50, 5000)) 
                out += new_num
                temp = ""
            out += c
    if temp != "":
        new_num = str(random.randint(50, 5000))
        out += new_num
    return out

def replace_locations(text: str):
    """Swaps known city names for random other cities."""
    if not isinstance(text, str): 
        return text
    
    fake_loc = random.choice(LOCATIONS)
    # Simple replace logic
    for loc in ["Hà Nội", "TP HCM", "Đà Nẵng", "Huế", "Cần Thơ"]:
        text = text.replace(loc, fake_loc)
    return text

# -----------------------------------------------------
# GENERATOR LOGIC
# -----------------------------------------------------

def generate_fake_variations(row):
    """
    Returns a LIST of dictionaries (multiple fake rows per real row).
    """
    variations = []
    
    # Base data
    base_title = row.get("title", "")
    base_content = row.get("content", "")
    
    if not isinstance(base_content, str):
        base_content = ""

    # ------------------------------------------
    # TYPE 1: NUMERIC DISTORTION ONLY
    # (Good for training models to spot statistical anomalies)
    # ------------------------------------------
    content_numeric = distort_number(base_content)
    if content_numeric != base_content: # Only add if changes happened
        variations.append({
            "id": f"{row['id']}_num",
            "url": fake_url(),
            "title": f"{base_title} (Sai số liệu)",
            "description": row.get("description"),
            "content": content_numeric,
            "label": "untrusted",
            "fake_type": "numeric_distortion",
            "scraped_at": row["scraped_at"],
            "published_date": row["published_date"],
        })

    # ------------------------------------------
    # TYPE 2: ENTITY/LOCATION SWAP ONLY
    # (Good for training models to spot factual inconsistencies)
    # ------------------------------------------
    content_loc = replace_locations(base_content)
    if content_loc != base_content:
        variations.append({
            "id": f"{row['id']}_loc",
            "url": fake_url(),
            "title": f"{base_title} (Sai địa điểm)",
            "description": row.get("description"),
            "content": content_loc,
            "label": "untrusted",
            "fake_type": "entity_swap",
            "scraped_at": row["scraped_at"],
            "published_date": row["published_date"],
        })

    # ------------------------------------------
    # TYPE 3: CLICKBAIT / EXAGGERATION
    # (Good for sentimental/tonal analysis)
    # ------------------------------------------
    fake_person = random.choice(FAKE_PEOPLE)
    exaggeration = random.choice(EXAGGERATION_PHRASES)
    prefix = random.choice(COUNTERFACTUAL_STARTERS)
    
    content_context = (
        f"{prefix} Theo lời kể của {fake_person}, {base_content}\n\n"
        f"Sự việc này đã {exaggeration}."
    )
    
    variations.append({
        "id": f"{row['id']}_ctx",
        "url": fake_url(),
        "title": f"SỐC: {base_title}",
        "description": row.get("description"),
        "content": content_context,
        "label": "untrusted",
        "fake_type": "clickbait_fabrication",
        "scraped_at": row["scraped_at"],
        "published_date": row["published_date"],
    })

    return variations

# -----------------------------------------------------
# MAIN PROCESS
# -----------------------------------------------------

def main():
    # 1. Load Data
    try:
        df = pd.read_csv("output.csv")
    except FileNotFoundError:
        print("Error: output.csv not found.")
        return

    all_fakes = []

    # 2. Generate multiple variations for each row
    for _, row in df.iterrows():
        fakes = generate_fake_variations(row)
        all_fakes.extend(fakes) # Use extend because function returns a list

    # 3. Create DataFrame
    df_fake = pd.DataFrame(all_fakes)
    
    # 4. Merge
    df_final = pd.concat([df, df_fake], ignore_index=True)

    # 5. Save
    df_final.to_csv("output_with_distinct_fakes.csv", index=False)
    print(f"✅ Done! Original rows: {len(df)}")
    print(f"✅ Generated fake rows: {len(df_fake)}")
    print(f"✅ Total rows saved: {len(df_final)}")

if __name__ == "__main__":
    main()