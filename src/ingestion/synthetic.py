import os
import random
import datetime
import pandas as pd

# Target templates and words to make the dataset highly realistic
PRODUCTS = ["Nexus Laptop", "Nova Smartwatch", "Aero Earbuds", "Apex Keyboard", "Zenith Tablet", "CloudHosting", "ExpressDelivery"]
USERS = ["alex_g", "sarah_m", "john_d", "tech_geek", "art_creator", "fit_runner", "movie_buff", "office_worker", "happy_customer", "audiophile_99", "gamer_pro", "buyer_first", "dev_ops", "finance_guru", "traveler_101", "design_master", "student_98", "blog_writer", "coffee_lover", "speed_demon"]
SOURCES = ["Amazon", "AppStore", "Google Play", "Support Ticket", "Email", "Web Forum"]

TEMPLATES = {
    "product_review": {
        "positive": [
            "The {product} is absolutely amazing! The {aspect} is top notch and I love the design.",
            "Highly recommend the new {product}. The {aspect} works flawlessly and it's great value for money.",
            "I've been using the {product} for a month now. Extremely satisfied with the {aspect} and overall performance.",
            "The {product} exceeded my expectations. The {aspect} is super responsive and clean."
        ],
        "negative": [
            "I bought the {product} last week and I am disappointed. The {aspect} is terrible and very slow.",
            "Do not buy the {product}! The {aspect} is buggy, unreliable, and crashes constantly.",
            "The {product} is overpriced. The {aspect} feels cheap and it is not user friendly.",
            "Worst purchase of the year. The {product} has major issues with the {aspect}."
        ]
    },
    "service_review": {
        "positive": [
            "Excellent customer service from the {product} team. They resolved my query quickly.",
            "Great support for {product}. The technicians were helpful and friendly.",
            "The onboarding for {product} was smooth. The client portal is very easy to use.",
            "I had a great experience with {product} logistics. Fast delivery and neat packaging."
        ],
        "negative": [
            "Terrible customer service for {product}. I've been waiting for a response for three days.",
            "The support team for {product} was unhelpful and kept transferring my call.",
            "Extremely slow turnaround time. They did not resolve my issues with {product}.",
            "Frustrated with {product} service. They promised a refund but I haven't received it yet."
        ]
    },
    "support_ticket": {
        "positive": [
            "Support ticket solved: My account credentials for {product} were restored successfully. Thanks!",
            "Thank you to the support desk for fixing my login issue on {product} so fast.",
            "Quick resolution of my support request for {product}. Excellent tech support.",
            "Appreciate the support team's guidance on setting up the {product} integration."
        ],
        "negative": [
            "Support ticket: The {product} application is failing to launch after the latest update. Help!",
            "My support ticket has been open for a week now. I cannot access my {product} workspace.",
            "Support issue: Cannot configure SSO for {product}. Getting a 500 internal server error.",
            "Critical issue: The database for {product} is showing high CPU usage and queries are timing out."
        ]
    },
    "feature_request": {
        "positive": [
            "Request: It would be great if {product} could add a dark mode option. Otherwise, love the app!",
            "Feature Request: Please add export to CSV functionality in the {product} reporting module.",
            "Can we get a mobile widget for {product}? That would make tracking metrics much easier.",
            "I would love to see an integration between {product} and Slack. It would boost our workflow."
        ],
        "negative": [
            "Why does {product} lack basic auto-save? I lost my work because it crashed. Please add it ASAP.",
            "The {product} search functionality is useless right now. We need advanced keyword filters.",
            "Requesting multi-language support for {product}. The app currently only works in English.",
            "We need custom role permissions for {product}. The current admin-only setup is too restrictive."
        ]
    },
    "bug_report": {
        "positive": [
            "Bug: Found a small display glitch in {product} settings, but refreshing the page fixes it.",
            "Formatting issue: The table headers in {product} dashboard overlap on safari, but chrome is fine.",
            "Minor bug: The email notification for {product} alert has a typo in the subject line.",
            "Interface glitch: The toggle button on {product} profile page takes two clicks to register."
        ],
        "negative": [
            "Bug Report: The {product} app crashes every time I upload a PDF file larger than 5MB.",
            "Critical bug: Saving settings in {product} throws a network error and loses changes.",
            "The latest patch for {product} broke the OAuth login. No one in our team can log in.",
            "Bug: The data export on {product} has missing columns and values are misaligned."
        ]
    },
    "billing_complaint": {
        "positive": [
            "Billing query: Just wanted to confirm that the discount was applied to my {product} invoice.",
            "The billing department refunded my double payment on {product} promptly. Thanks.",
            "Billing issue resolved: Got a credit note for the downtime on {product}.",
            "Invoice received for {product} renewal, details look correct."
        ],
        "negative": [
            "Billing Complaint: I was charged twice for my monthly {product} subscription. Refund me!",
            "Overcharged on {product} bill! The invoice does not match the quoted price.",
            "My credit card was charged for {product} even though I cancelled my account last month.",
            "Billing issue: The invoice PDF for {product} cannot be downloaded, says file corrupted."
        ]
    }
}

ASPECTS = {
    "UI": ["user interface", "UI layout", "screen styling", "dashboard view", "buttons", "visual theme", "menus"],
    "Battery": ["battery life", "power usage", "charging port", "battery health", "power backup", "power consumption"],
    "Performance": ["processing speed", "loading times", "system response", "crashes", "lag", "reliability", "CPU usage"],
    "Pricing": ["cost", "subscription price", "value for money", "billing model", "pricing tiers", "renewal cost"],
    "Support": ["customer support", "help desk", "tech assistance", "service turn-around", "support responses"],
    "Security": ["data privacy", "SSO integration", "login safety", "encryption", "user permissions", "auth tokens"],
    "Features": ["options", "features set", "export formats", "dark mode toggle", "integrations", "tools library"]
}

def generate_synthetic_dataset(output_path: str = "data/raw/synthetic_dataset.csv", count: int = 50000) -> str:
    """
    Generates a realistic synthetic customer review dataset and saves it to a CSV file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    categories = list(TEMPLATES.keys())
    sentiments = ["positive", "negative"]
    
    records = []
    
    # Generate mock reviews
    start_date = datetime.datetime.now() - datetime.timedelta(days=365)
    
    for i in range(count):
        category = random.choice(categories)
        sentiment = random.choice(sentiments)
        
        # Select product
        product = random.choice(PRODUCTS)
        
        # Select aspect
        aspect_key = random.choice(list(ASPECTS.keys()))
        aspect_term = random.choice(ASPECTS[aspect_key])
        
        # Build text
        template = random.choice(TEMPLATES[category][sentiment])
        text = template.format(product=product, aspect=aspect_term)
        
        # Metadata
        user = random.choice(USERS)
        source = random.choice(SOURCES)
        
        # Distribute timestamps over the last year
        random_days = random.randint(0, 365)
        random_hours = random.randint(0, 23)
        random_minutes = random.randint(0, 59)
        timestamp = start_date + datetime.timedelta(days=random_days, hours=random_hours, minutes=random_minutes)
        
        rating = float(random.randint(4, 5)) if sentiment == "positive" else float(random.randint(1, 2))
        if random.random() < 0.1: # Add some neutral/mixed ratings
            rating = 3.0
            
        records.append({
            "text": text,
            "category": category.replace("_", " ").title(),
            "sentiment": sentiment.upper(),
            "rating": rating,
            "product": product,
            "user": user,
            "source": source,
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S")
        })
        
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} synthetic records and stored at: {output_path}")
    return output_path

if __name__ == "__main__":
    # Test generation of a small batch
    generate_synthetic_dataset(output_path="data/raw/test_synthetic.csv", count=100)
    print("Done")
