import faiss
import numpy as np
from openai import OpenAI

# Initialize OpenAI Client (Adjust base_url for local Ollama if needed)
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

# Shipping and Return Policy Text
policy_data = [
    "SHIPPING INFORMATION\nWe ship to a vast network of international addresses, ensuring you receive your order almost anywhere. However, some product restrictions and destination limitations may apply.",
    "When you place an order, we will estimate your shipping and delivery dates based on the availability of your items and the shipping options you choose.",
    "SHIPPING RATES\nWe offer a flat shipping rate on all orders that meet specific criteria. This means no matter how many items you order that fall within these limits, you'll pay the exact shipping cost. It's a simple and affordable way to get your items delivered.",
    "$0.01-$100.00 - $12.95 Flat Shipping Rate",
    "$100.01-$200.00 - $17.95 Flat Shipping Rate",
    "$200.01-$300.00 - $22.95 Flat Shipping Rate",
    "$300.01-$499.00 - $25.00 Flat Shipping Rate",
    "$500.00+ - Free Ground Shipping",
    "For international orders, please get in touch with Customer Service at (732) 447-1102.",
    "*Exclusions: Flat Rates and Free Shipping are not available on international and special orders, including non-stock items, reagents, and chemicals.",
    "RETURN POLICY\nWe want you to be happy with your purchase! You can return most new, unopened items within 30 days of delivery for a full refund. We cover return shipping if the mistake (incorrect or defective item) is on our end.",
    "Some exceptions apply:\nHazardous materials, certain consumables, flammable items, and non-returnable items may have restocking fees or be excluded from returns altogether. These items will be marked on the product page.",
    "ABC Enterprises reserves the right to designate specific items as non-returnable.",
    "We recommend checking the product page for any return restrictions before you buy. For more details on our return policy, please visit our Terms & Conditions page.",
    "STARTING THE RETURN PROCESS\nRETURN APPROVALS\nFor a smooth return experience, we recommend initiating your return request within 30 days of receiving your order.",
    "To request a return, contact our Customer Service Department at (732) 447-1102 within 30 days of receiving your order. To expedite your call, please have the following information ready:\nOrder Number\nContact Information\nItem(s) Numbers\nReturn Reason",
    "Manufacturer authorization (if applicable): In some cases, certain items may require manufacturer approval before we can finalize your return. We'll inform you if this applies to your specific item and guide you through the process.",
    "Return authorization (if approved): Once approved (by ABC Enterprises and/or the manufacturer, if applicable), you'll receive a return authorization within 7 calendar days of your initial request. This authorization allows you to return the item.",
    "You will receive from ABC Enterprises a valid RMA (Return Merchandise Authorization) and a shipping return label. Returns will not be accepted without an RMA.",
    "RESTOCKING FEES\nWe strive to ensure a smooth return experience. However, a restocking fee may apply to certain returns to cover the costs of restocking the item.",
    "Non-faulty Returns: A minimum 25% restocking fee may apply.",
    "Manufacturer Fees: In some cases, the manufacturer may have a higher restocking fee that we'll need to pass on. The specific fee will be clearly communicated at the time of return.",
    "For your reference:\nRestocking fees typically apply to returns, not due to our error (incorrect or defective item).",
    "You can check the product page or our returns policy for any exceptions or specific fees associated with an item.",
    "SHIPPING YOUR RETURNS\nTo ensure a smooth return, please make sure authorized item(s) are:\nIn their original, complete packaging. This includes all boxes, manuals, and any other materials that came with the product.",
    "In original condition. This means the item(s) should be unused and undamaged, with all components present.",
    "If the item(s) you received were packaged in dry ice or frozen, it must be returned in the same manner.",
    "Packaged items should be returned to:\nABC Enterprises\nRMA# [Insert No.]\n121 Jersey Avenue\nNew Brunswick, NJ 08901",
    "Ensure that your RMA # is marked clearly on the return shipping address. Shipping charges will not be reimbursed. To ensure your return arrives safely and your refund is processed quickly, please use the original packaging. Be sure to pack the item securely with sufficient padding to prevent damage during transit. ABC Enterprises will not be responsible for damage caused by shipping returns.",
    "GETTING YOUR REFUND\nOnce your return has been approved and processed, your refund will be returned to the original form of payment. Credits returned to the original form of payment may vary by your bank’s institution. All P.O.s will be credited after our returns department inspects and accepts your return. While we aim for a speedy return process, you should receive your refund within four weeks of returning your package. This timeframe includes:\nReturn shipping time (5-10 business days)\nOur processing time (3-5 business days)\nYour bank's processing time (5-10 business days)",
    "If you do not receive a credit after 30 days of ABC Enterprises accepting your return, please call our Customer Service Department at (732) 447-1102.",
    "LOST PACKAGE CLAIM\nWhile we understand the importance of getting your order, unfortunately, lost packages happen occasionally. Please call your return shipping carrier.",
    "Be sure to provide the necessary documentation (tracking number, shipment details) to assist in filing a claim with the shipping company"
]

# Convert policy data into embeddings
def get_embedding(text):
    response = client.embeddings.create(
        model="all-minilm",
        input=text
    )
    return response.data[0].embedding

# Generate embeddings for policy data
policy_embeddings = np.array([get_embedding(text) for text in policy_data]).astype('float32')

# Create FAISS index
embedding_size = len(policy_embeddings[0])
index = faiss.IndexFlatL2(embedding_size)
index.add(policy_embeddings)

# Save index
faiss.write_index(index, "policy_index.faiss")

# Store the original policy text
with open("policy_texts.txt", "w") as f:
    for text in policy_data:
        f.write(text + "\n")

print("Policy data indexed successfully!")
