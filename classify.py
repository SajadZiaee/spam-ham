import torch
import torch.nn as nn
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
import re
import pandas as pd


class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.sigmoid(self.fc5(x))
        return x


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'\d+', 'num', text)  # Replace numbers with 'num'
    # Replace multiple whitespaces with a single whitespace
    text = re.sub(r'\s+', ' ', text)
    return text


def classify_text(text):
    preprocessed_text = preprocess_text(text)

    # Load the dataset and preprocess it
    data = pd.read_csv('spam_or_not_spam.csv')
    data = data.dropna()
    X = data['email'].values

    # Fit the CountVectorizer on the dataset
    vectorizer = CountVectorizer()
    vectorizer.fit(X)

    preprocessed_text = vectorizer.transform([preprocessed_text])
    preprocessed_text = csr_matrix(
        preprocessed_text).todense()  # Convert to dense tensor

    input_dim = preprocessed_text.shape[1]
    model = Net(input_dim)
    model.load_state_dict(torch.load('spam_detection_model.pt'))
    # Convert to torch.Tensor
    output = model(torch.tensor(preprocessed_text, dtype=torch.float32))
    print(output.item())
    classification = 'Spam' if output.item() >= 0.5 else 'Ham'
    return classification


# Example usage
spam1 = """Dear Subscriber,

This message is brought to you by [Company Name] in compliance with current federal laws. To find out more about [Company Name] and the offers we provide, please visit our website: [Website URL].

You are receiving this email because you or someone you know has registered this email address to receive special offers from an [Company Name] marketing partner. We have carefully screened the email addresses to the best of our knowledge. However, if you believe you have received this email in error, we apologize for any inconvenience it has caused. Rest assured that we will not send any further offers to you.

If you wish to be unlisted from our database, please follow the simple steps below:

    Click on the following link: [Unsubscribe Link]
    If your mail has been forwarded to a new email address, please provide us with your old email address to ensure successful unlisting.
    For any questions or support, please contact our nationwide support team at [Support Email] or visit [Support Website].

At [Company Name], we take pride in our e-mail campaigns that have produced staggering response rates. Whether you require responsive general or targeted managed e-mail lists, we are here to assist you. Visit our website [Website URL] today to explore the options.

Copyright [Current Year] [Company Name]. All rights reserved.

Thank you for your attention to this matter.

Best regards,
Special Offers Team"""
classification = classify_text(spam1)
print(f"Classification Spam 1: {classification}")


spam2 = """Dear Valued Customer,

Here is an excerpt from your local newspaper featuring an interview with a curious computer user:

Q: "Is my computer supposed to run this slow?"
A: "No, your computer should be as fast as the day you purchased it. The solution to your problem is Norton SystemWorks [VERSION NUMBER]."

Q: "I think I have a virus. What do I do?"
A: "Act quickly before the virus spreads and infects your entire system. You must get a copy of Norton SystemWorks [VERSION NUMBER]."

Q: "I am worried that I may lose my data if my computer crashes. How do I backup my data safely and easily?"
A: "Everything you need for data backup is included in Norton SystemWorks [VERSION NUMBER]."

Q: "I occasionally need to send a fax with my computer. What will make this easier for me?"
A: "WinFax, the easiest-to-use fax software available, is also included in Norton SystemWorks [VERSION NUMBER]."

Q: "This SystemWorks [VERSION NUMBER] sounds like it does a lot for my computer. Can anyone use this software?"
A: "Yes, it is easy to use, and tech support is included. Norton SystemWorks [VERSION NUMBER] is the best software available on the market, ensuring you and your PC have a better relationship."

Q: "Okay, but wait. It must cost a ton of money, right?"
A: "Well, usually yes, but this is a special offer. It sells at your local computer store for [PRICE]. However, it is available for a limited time for only [DISCOUNTED PRICE] with free shipping."

Q: "What a deal! So, how do I order?"
A: "To order, please click on the following link: [Order Link]"

Q: "Great, thanks!"
Q: "One more question, how do I get removed from this email list?"
A: "That is not a problem. Please click on the following link to be removed within the legal period of [NUMBER] business days: [Unsubscribe Link]"

Thank you for your interest in Norton SystemWorks [VERSION NUMBER]. We strive to enhance your computer's performance and ensure a secure computing experience.

Best regards,

Norton SystemWorks Team"""
classification = classify_text(spam2)
print(f"Classification Spam 2: {classification}")


spam3 = """Dear Homeowner,

Are you looking for the best mortgage rates to jump-start your plans for the future? Look no further! We can help you find the perfect mortgage solution by connecting you with hundreds of lenders who will compete for your loan.

Whether you are interested in refinancing, new home loans, debt consolidation, second mortgages, or home equity loans, our service is designed to match your needs with the best options available in the market.

Why choose us?

    Simple and Easy Process: Our streamlined process makes finding the best mortgage rates hassle-free. No complex forms or lengthy procedures!

    Free Service: Our service is completely free for homeowners and new home buyers. There are no hidden fees or obligations.

    Extensive Lender Network: We have partnered with hundreds of lenders, ensuring that you have access to a wide range of options.

    Less Than Perfect Credit? No Problem: Even if you have less than perfect credit, you are still eligible to explore the best mortgage rates that suit your situation.

Don't miss out on this opportunity to secure the lowest interest rates in [NUMBER] years. Take the first step towards your future plans today by filling out a quick and simple form.

To get started, please click on the following link: [Start Your Mortgage Search]

We respect your privacy and value your time. You are receiving this email because you registered at one of URL's partner sites and agreed to receive gifts and special offers. If you no longer wish to receive such offers in the future, please click on the following link to unsubscribe: [Unsubscribe Link]

Thank you for considering Mortgage Match as your trusted partner in finding the best mortgage rates. We are committed to helping you achieve your homeownership goals.

Sincerely,

The Mortgage Match Team"""
classification = classify_text(spam3)
print(f"Classification Spam 3: {classification}")


ham1 = """Hi,

I recently came across Solaris and Linux servers and had a question regarding their differences. I've been using Linux for some time now, but I'm not familiar with Solaris. Can anyone explain the key distinctions between these two operating systems? I'm considering getting a server, and I want to make an informed decision.

Thanks,
John"""
classification = classify_text(ham1)
print(f"Classification Ham 1: {classification}")


ham2 = """Hello,

I recently came across an intriguing concept called LifeGem diamonds, and I wanted to learn more about this unique memorial option. It's fascinating how they create certified high-quality diamonds from the carbon of a loved one as a way to remember their remarkable life. However, I have a question that crossed my mind: Is it limited to human remains, or could other carbon sources be used as well?

I apologize if this question sounds unusual, but I'm genuinely curious about the process and potential possibilities. Thank you for any insights you can provide.

Warm regards,
Emily"""
classification = classify_text(ham2)
print(f"Classification Ham 2: {classification}")


ham3 = """Hello,

I hope this email finds you well. I wanted to share some exciting news regarding the use of Linux in education. The Irish Linux Users Group (ILUG) recently established a marketing special interest group, which has already sparked an intriguing initiative at University College Cork (UCC).

Starting this academic year, every incoming student at UCC will be offered a complimentary copy of Red Hat Linux VERSION, thanks to the proposal made by ILUG member Braun Brelin. Braun, who serves as the Director of Technology at OpenApp, ran a training class for staff at the UCC Computer Science Department, where he introduced the idea of providing Linux to students. This initiative could potentially extend to other Irish universities as well.

ILUG is collaborating with an international Red Hat program aimed at introducing students to open-source computing. Red Hat, the Linux distributor, offers educational software bundled with its operating environment and provides networked support services to eligible applicants. While this program was initially designed for the US educational system, it is now available to schools and universities worldwide.

Red Hat Linux VERSION emphasizes ease of use and maintenance, addressing the perception that Linux can be challenging to master on personal systems. The "Linux for All" project at UCC not only benefits the students but also raises the profile of Red Hat Ireland, based in Cork. Red Hat Ireland has been providing shared financial services for other Red Hat offices in Europe since NUMBER. With the ILUG marketing group's formation, David Owens, Red Hat's Director of Global Logistics and Production, sees an opportunity to adopt a more proactive approach. The office in Cork has received an increasing number of calls from Irish companies interested in adopting Linux, and they have introduced some to Red Hat's pre-sales consultants.

I would like to express my appreciation to Braun and David for their collaborative efforts in making this initiative a reality. It is an exciting time for Linux in education, and we look forward to exploring the endless possibilities it offers.

If you have any questions or would like further information, please don't hesitate to reach out. Thank you for your time and support.

Warm regards,
Michael Johnson"""
classification = classify_text(ham3)
print(f"Classification Ham 3: {classification}")


ham4 = """Hello there,

This is a friendly message from Fork Admin, and we hope you're having a great day! We wanted to address a recent conversation thread and share some thoughts with you.

Firstly, let's talk about the subject at hand. We appreciate the humor and wit displayed in the discussion, and it's always refreshing to engage in lighthearted banter. It seems that the topic of anime and its influence on our lives has sparked some interest. While everyone has their own preferences and passions, we can all agree that anime has a unique way of capturing our imagination and taking us on incredible journeys.

By the way, kudos to those who guessed the anime reference related to green coolant in Navi! It's these little nods and shared experiences that bring us together. Anime has a way of leaving a lasting impression and creating connections, and we hope you continue to enjoy exploring its diverse world.

On a different note, we wanted to remind you of the importance of respecting one another's opinions and perspectives. Our community thrives when we foster an environment of inclusivity and understanding. We encourage open discussions while maintaining a level of respect and kindness towards fellow members.

At Fork, we value the diversity of thoughts and ideas that each individual brings to the table. It's this richness that makes our community vibrant and engaging. So let's keep the conversations going, share our passions, and discover new perspectives along the way.

If you have any questions, concerns, or simply want to chat about your latest anime recommendations, feel free to reach out to us. We're here to make your experience at Fork enjoyable and fulfilling.

Thank you for being a valued member of our community!

Best regards,

The Fork Admin Team"""
classification = classify_text(ham4)
print(f"Classification Ham 4: {classification}")
