import pandas as pd


# Build a balanced dataset with realistic variations of customer messages.
data = {
    "shipping": [
        "My package has not arrived yet.",
        "Where can I track my order?",
        "My shipment is stuck in transit.",
        "The courier says delivered but I do not have it.",
        "My delivery is late.",
        "When will my package arrive?",
        "My order has not been delivered.",
        "The tracking information has not updated.",
        "Can you tell me where my package is?",
        "My parcel was supposed to arrive yesterday.",
        "The delivery date has passed.",
        "My package is missing.",
        "Why is my shipment delayed?",
        "The courier has not delivered my order.",
        "My order is still in transit.",
        "I have not received my delivery.",
        "Can I get an update on my shipment?",
        "The tracking number is not showing any updates.",
        "My package was marked delivered but it is missing.",
        "How long will shipping take?",
        "Where is my parcel?",
        "My order is taking too long to arrive.",
        "The package has been delayed again.",
        "I am still waiting for my delivery.",
        "My shipment has not moved for days.",
        "The tracking status has not changed.",
        "My parcel has been in transit for too long.",
        "Why has my delivery not arrived?",
        "Can you check the status of my package?",
        "My order was supposed to arrive today.",
        "The courier has not shown up.",
        "I cannot find my delivery.",
        "My package appears to be lost.",
        "The delivery is overdue.",
        "I need an update about my order delivery.",
        "My shipment is delayed.",
        "The tracking page says my package is still in transit.",
        "My package has been stuck since last week.",
        "I am waiting for a package that should have arrived.",
        "Can you tell me when my order will be delivered?",
        "The courier has not delivered my parcel.",
        "My delivery status has not changed.",
        "My order has been shipped but has not arrived.",
        "I have been waiting for my package all week.",
        "The tracking number is not working.",
        "Why is my package taking so long?",
        "My parcel never arrived.",
        "The delivery was marked complete but I received nothing.",
        "I need help tracking my shipment.",
        "Where is my order right now?",
        "My delivery is overdue.",
        "The package has disappeared during shipping.",
        "My shipment has not updated since yesterday.",
        "I expected my package already.",
        "Can someone check my delivery status?",
        "My order is still waiting to be delivered.",
        "The courier says my package was delivered, but it was not.",
        "My parcel has been delayed for several days.",
        "How can I find my missing package?",
        "My delivery has not arrived as expected.",
    ],
    "account": [
        "I forgot my password.",
        "I cannot log into my account.",
        "My account has been locked.",
        "How do I change my email address?",
        "I cannot access my account.",
        "How can I reset my password?",
        "I need to update my account details.",
        "Why is my account locked?",
        "I forgot my login details.",
        "I am unable to sign in.",
        "How do I change my account information?",
        "My password reset is not working.",
        "I cannot access my profile.",
        "How do I update my phone number?",
        "My login is not working.",
        "I need help recovering my account.",
        "How can I change my password?",
        "My account access has been blocked.",
        "I cannot remember my password.",
        "I need to update my personal information.",
        "I am locked out of my account.",
        "My login credentials are not working.",
        "How do I recover my account?",
        "I need to reset my login.",
        "I cannot sign into my profile.",
        "My password is not being accepted.",
        "How can I unlock my account?",
        "I want to change my account email.",
        "I need to update my profile.",
        "My account is inaccessible.",
        "I forgot the password I use to log in.",
        "How do I change my phone number?",
        "I cannot remember my login information.",
        "My account was locked after several attempts.",
        "I need help changing my password.",
        "I cannot get into my account.",
        "How can I update my personal details?",
        "My sign-in is failing.",
        "I need to recover my login.",
        "Why can I not access my profile?",
        "I want to update my email.",
        "My password reset link is not working.",
        "How do I unlock my profile?",
        "I lost access to my account.",
        "My account login keeps failing.",
        "Can I change the email associated with my account?",
        "I need to recover my password.",
        "My account is currently locked.",
        "I cannot sign in with my password.",
        "How do I edit my account information?",
        "I need help getting back into my account.",
        "My login details are no longer working.",
        "How do I reset my account password?",
        "I cannot access my profile anymore.",
        "My account credentials are not working.",
        "I need to change my personal details.",
        "I forgot how to log in.",
        "The system will not let me sign in.",
        "Can you help me recover my account?"
        "I am having trouble accessing my account.",
    ],
    "technical": [
        "The application keeps crashing.",
        "The website will not load.",
        "I keep getting an error.",
        "The dashboard is frozen.",
        "The app stopped working.",
        "The website keeps showing an error.",
        "A feature is not working correctly.",
        "The application crashes when I open it.",
        "The page is stuck loading.",
        "I cannot use the search feature.",
        "The system is responding very slowly.",
        "The app keeps freezing.",
        "The website is broken.",
        "I am seeing an unexpected error.",
        "The dashboard is not loading.",
        "The application stopped responding.",
        "A button on the website does not work.",
        "The system keeps timing out.",
        "The app is not functioning properly.",
        "I cannot use the platform because it keeps crashing.",
        "The website keeps crashing.",
        "The app will not open.",
        "I cannot load the dashboard.",
        "The page keeps freezing.",
        "The system keeps displaying errors.",
        "The application is extremely slow.",
        "The website is stuck on the loading screen.",
        "A feature suddenly stopped working.",
        "The app crashes every time I open it.",
        "The website is not responding.",
        "The system keeps timing out.",
        "The dashboard stopped working.",
        "I cannot get the application to start.",
        "The page is completely frozen.",
        "The website gives me an error message.",
        "The application is not responding.",
        "I am having technical problems with the platform.",
        "The search function is broken.",
        "The website keeps going down.",
        "The app freezes when I try to use it.",
        "The dashboard will not open.",
        "I cannot access a feature on the website.",
        "The application keeps giving me errors.",
        "The website takes forever to load.",
        "Something is wrong with the application.",
        "The system is not working properly.",
        "The page keeps crashing.",
        "I cannot use the dashboard.",
        "The app stopped responding.",
        "The website is malfunctioning.",
        "I keep seeing an error when using the platform.",
        "The application will not respond.",
        "The system has become very slow.",
        "The website does not work on my device.",
        "The app is stuck.",
        "The dashboard keeps freezing.",
        "The platform is showing an unexpected error.",
        "A page will not load correctly.",
        "The application crashes whenever I use it.",
        "The website is not functioning properly."
    ],
    "general_support": [
        "I need help with my order.",
        "Can someone help me with this issue?",
        "I have a question about my service.",
        "I need assistance with my request.",
        "Can you explain how this works?",
        "I would like to make a request.",
        "I need help resolving an issue.",
        "Can someone from support assist me?",
        "I have a question for customer service.",
        "I would like more information.",
        "I am unhappy with the service I received.",
        "I have contacted support but still need help.",
        "Can you help me with this problem?",
        "I need someone to look into this.",
        "I would like to speak with support.",
        "I have a problem that needs to be resolved.",
        "Please help me resolve this issue.",
        "I am not satisfied with the service.",
        "I need assistance from customer support.",
        "Can you help me understand what I should do?",
        "I need some assistance.",
        "Can someone help me?",
        "I have an issue and need support.",
        "I need to contact customer service.",
        "Can you help me with my request?",
        "I have a general question.",
        "I need help understanding something.",
        "Please assist me with this problem.",
        "I would like to contact support.",
        "Can someone look into this for me?",
        "I need customer service assistance.",
        "I have a problem with my service.",
        "Can you help me figure this out?",
        "I need someone from support to help me.",
        "I would like some assistance with this.",
        "I need help with a problem.",
        "Can you tell me what I should do?",
        "I have an issue I need help with.",
        "Please have someone assist me.",
        "I need more information about my request.",
        "Can customer support help me?",
        "I would like to ask a question.",
        "I need assistance from someone on your team.",
        "Can someone explain this to me?",
        "I am having an issue and need help.",
        "I need support with this situation.",
        "Please help me with my request.",
        "I want to speak to customer service.",
        "Can you look into my issue?",
        "I need help resolving this.",
        "I have a concern I would like addressed.",
        "Can someone from your team assist me?",
        "I need clarification about something.",
        "I am not sure what to do next.",
        "Could someone help me with this?",
        "I need assistance from support.",
        "I have a problem that I need help solving.",
        "Please tell me how I can get help.",
        "I would like someone to review my issue.",
        "Can support help me resolve this?"
    ]
}


# Convert the category messages into rows for the training dataset.
rows = []

for category, messages in data.items():
    for message in messages:
        rows.append(
            {
                "text": message,
                "category": category
            }
        )


# Create the dataframe from all training examples.
df = pd.DataFrame(rows)


# Balance the dataset using the same number of examples from each category.
category_counts = df["category"].value_counts()
minimum_examples = category_counts.min()

df = (
    df.groupby("category", group_keys=False)
    .head(minimum_examples)
    .reset_index(drop=True)
)

print("Balanced dataset:")
print(df["category"].value_counts())

# Save the generated dataset for model training.
df.to_csv(
    "data/training_data.csv",
    index=False
)


print(
    f"Created dataset with {len(df)} records."
)

print()

print(
    df["category"].value_counts()
)