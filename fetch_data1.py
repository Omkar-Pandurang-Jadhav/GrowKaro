import googlemaps
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import time
import json
from google import genai

# ------------------ Setup ------------------
API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY or not GEMINI_KEY:
    print("Error: Please set GOOGLE_MAPS_API_KEY and GEMINI_API_KEY in environment variables.")
    sys.exit(1)

gmaps = googlemaps.Client(key=API_KEY)
client = genai.Client(api_key=GEMINI_KEY)

# ------------------ User Input ------------------
location_input = input("Enter city/location (e.g., Mumbai CST): ").strip()
business_type = input("Enter type of business (e.g., restaurant, electronics_store): ").strip().lower()

# ------------------ Gemini Review Analysis ------------------
def analyze_reviews_batch(review_texts, retries=3):
    """
    Uses Gemini API to detect aspects and sentiment for multiple reviews at once.
    Returns a list of dicts [{aspect: sentiment}, ...] corresponding to review_texts.
    """
    batch_results = []
    for review_text in review_texts:
        for attempt in range(retries):
            try:
                prompt = f"""
                You are a review analysis assistant.
                Extract key aspects mentioned in the review below and classify the sentiment
                (positive, negative, neutral) for each aspect.
                Return a valid JSON object only, where keys are aspects and values are sentiments.

                Review: "{review_text}"
                """
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[prompt]
                )
                text = response.text.strip()
                if text:
                    batch_results.append(json.loads(text))
                    break
                else:
                    print(f"Empty response, retry {attempt + 1}")
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(1)  # wait before retry
        else:
            batch_results.append({})  # empty dict if all retries fail
    return batch_results

# ------------------ Main Logic ------------------
try:
    # Geocode location
    geocode_result = gmaps.geocode(location_input)
    if not geocode_result:
        print("Location not found.")
        sys.exit(1)

    lat = geocode_result[0]["geometry"]["location"]["lat"]
    lng = geocode_result[0]["geometry"]["location"]["lng"]
    location = f"{lat},{lng}"

    # Search nearby businesses
    places_result = gmaps.places_nearby(
        location=location,
        radius=2000,
        type=business_type,
        keyword=business_type
    )

    results = places_result.get("results", [])
    if not results:
        print(f"No places found for '{business_type}' near '{location_input}'.")
        sys.exit(1)

    # Collect business info + reviews
    business_data = []
    reviews_data = []

    for place in results:
        place_id = place["place_id"]
        name = place.get("name", "N/A")
        address = place.get("vicinity", "N/A")
        rating = place.get("rating", None)
        reviews_count = place.get("user_ratings_total", 0)

        business_data.append({
            "Business": name,
            "Rating": rating,
            "Reviews_Count": reviews_count,
            "Address": address
        })

        # Fetch reviews (pagination)
        next_page_token = None
        while True:
            if next_page_token:
                details = gmaps.place(place_id=place_id, fields=["reviews"], page_token=next_page_token)
            else:
                details = gmaps.place(place_id=place_id, fields=["reviews"])

            reviews = details.get("result", {}).get("reviews", [])
            for review in reviews:
                reviews_data.append({
                    "Business": name,
                    "Review": review.get("text", ""),
                    "Rating": review.get("rating", None),
                    "Address": address,
                    "Reviews_Count": reviews_count
                })

            next_page_token = details.get("next_page_token", None)
            if not next_page_token:
                break
            time.sleep(2)

    # Save competitor data
    df_business = pd.DataFrame(business_data)
    df_business.to_csv("competitor_data.csv", index=False)
    print("\n--- Competitor Data ---")
    print(df_business)
    print("\n✅ Data saved to competitor_data.csv")

    # Reviews vs Rating graph
    plt.figure(figsize=(10, 6))
    plt.scatter(df_business["Reviews_Count"], df_business["Rating"], s=100, alpha=0.7, c="blue", edgecolors="k")
    for i, row in df_business.iterrows():
        plt.text(row["Reviews_Count"] + 2, row["Rating"] + 0.02, row["Business"], fontsize=8)
    plt.xlabel("Number of Reviews")
    plt.ylabel("Average Rating")
    plt.title(f"Competitor Quadrant: {business_type.capitalize()} near {location_input}")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()

    # Aspect-Based Sentiment Analysis
    if not reviews_data:
        print("No reviews found for these businesses.")
        sys.exit(1)

    df_reviews = pd.DataFrame(reviews_data)
    aspect_results = []

    print("\nRunning Aspect-Based Sentiment Analysis using Gemini API...")
    for business in df_reviews["Business"].unique():
        business_reviews = df_reviews[df_reviews["Business"] == business]["Review"].tolist()
        print(f"\nProcessing reviews for '{business}'... ({len(business_reviews)} reviews)")

        # Batch call for reviews
        batch_aspects = analyze_reviews_batch(business_reviews)
        for aspects_sentiments in batch_aspects:
            for aspect, sentiment in aspects_sentiments.items():
                aspect_results.append({
                    "Business": business,
                    "Aspect": aspect,
                    "Sentiment": sentiment
                })

    # Aspect Summary
    if aspect_results:
        df_aspect = pd.DataFrame(aspect_results)
        aspect_summary = df_aspect.groupby(["Business", "Aspect", "Sentiment"]).size().unstack(fill_value=0)
        aspect_summary_percent = aspect_summary.div(aspect_summary.sum(axis=1), axis=0) * 100

        print("\n--- Aspect-Based Sentiment (%) ---")
        print(aspect_summary_percent.round(2))

        # Plot each business
        for business in df_reviews["Business"].unique():
            if business in aspect_summary_percent.index:
                business_data = aspect_summary_percent.loc[business]
                business_data.plot(kind="bar", stacked=True, figsize=(10, 5), colormap="RdYlGn")
                plt.ylabel("Percentage (%)")
                plt.title(f"Aspect-Based Sentiment for '{business}'")
                plt.show()
    else:
        print("No aspect sentiment detected.")

except Exception as e:
    print(f"An unexpected error occurred: {e}")
