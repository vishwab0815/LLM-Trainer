import json
import random
import os

def generate_large_dataset(output_path: str = "dataset/intents_large.json", target_examples_per_intent: int = 15):
    """
    Generates a scaled-down synthetic dataset by expanding base templates.
    7 intents * 15 examples = 105 examples
    """
    
    # Define our base entities (expanded for variety)
    entities = {
        "location": [
            "Delhi", "Mumbai", "Chennai", "Bangalore", "Hyderabad", "Kolkata", "Pune",
            "London", "New York", "Paris", "Tokyo", "Dubai", "Singapore", "Sydney",
            "Chicago", "Los Angeles", "Toronto", "Berlin", "Rome", "Amsterdam"
        ],
        "date": [
            "tomorrow", "today", "next week", "this weekend", "Monday", "Friday",
            "Jan 15", "December 25th", "next month", "in 2 days", "tonight",
            "next Tuesday", "this afternoon", "next year", "ASAP"
        ],
        "food_item": [
            "pizza", "biryani", "burger", "sandwich", "pasta", "sushi", "tacos",
            "salad", "steak", "noodles", "fried rice", "dosa", "idli", "paneer tikka",
            "french fries", "chicken wings", "pad thai", "curry", "ramen"
        ],
        "order_id": [
            "12345", "98765", "A-4521", "B-9982", "ORD-123", "ORD-999", "77889"
        ]
    }

    # Template structures for each intent
    templates = {
        "greeting": [
            "hello", "hi", "good morning", "hey there", "greetings", "howdy",
            "what's up", "good evening", "good afternoon", "hi bot", "hey",
            "anybody there", "hola", "yo"
        ],
        "book_flight": [
            "book a flight to {location}",
            "I want to go to {location} {date}",
            "reserve a ticket to {location}",
            "get me a flight to {location}",
            "need a flight to {location} for {date}",
            "find flights heading to {location} {date}",
            "can you book a trip to {location}",
            "looking for airfare to {location}",
            "schedule my flight to {location} {date}",
            "I need to fly out to {location} {date}"
        ],
        "order_food": [
            "order me a {food_item}",
            "I want to eat {food_item}",
            "get me a {food_item}",
            "I want a {food_item}",
            "can I get some {food_item} please",
            "deliver a {food_item} to my house",
            "I'm craving {food_item}",
            "place an order for {food_item}",
            "bring me {food_item}",
            "I would like to order {food_item}"
        ],
        "check_weather": [
            "what is the weather in {location}",
            "is it raining in {location}",
            "weather forecast for {location}",
            "how is the weather in {location} {date}",
            "will it rain {date} in {location}",
            "temperature in {location} {date}",
            "do I need an umbrella in {location} {date}",
            "is it sunny in {location}",
            "show me the weather for {location}",
            "climate info for {location}"
        ],
        "cancel_order": [
            "cancel my order {order_id}",
            "I want to cancel order {order_id}",
            "please stop my order",
            "can you cancel order {order_id} please",
            "I changed my mind, cancel my {food_item} order",
            "abort order {order_id}",
            "do not deliver order {order_id}",
            "I need to cancel my purchase",
            "refund my order {order_id}"
        ],
        "track_order": [
            "where is my order {order_id}",
            "track order {order_id}",
            "when will my {food_item} arrive",
            "status of order {order_id}",
            "is my order {order_id} on the way",
            "how long for delivery",
            "check the status of {order_id}"
        ],
        "talk_to_human": [
            "I need to talk to a person",
            "connect me to an agent",
            "customer service please",
            "I want a human",
            "let me speak to a representative",
            "speak to someone real",
            "human support",
            "get me a live agent"
        ]
    }

    # Function to recursively multiply lists to create variations
    # We will use random choice to quickly build 2500 unique strings per intent
    
    generated_intents = []
    
    print(f"Generating dataset... Target: {target_examples_per_intent * len(templates)} items")
    
    for intent_name, intent_templates in templates.items():
        examples_set = set() # Use a set to ensure unique examples
        
        max_attempts = target_examples_per_intent * 10
        attempts = 0
        # Keep generating until we hit the target size or run out of permutations
        while len(examples_set) < target_examples_per_intent and attempts < max_attempts:
            attempts += 1
            template = random.choice(intent_templates)
            
            # Fill in the blanks if they exist
            sentence = template
            if "{location}" in sentence:
                sentence = sentence.replace("{location}", random.choice(entities["location"]))
            if "{date}" in sentence:
                sentence = sentence.replace("{date}", random.choice(entities["date"]))
            if "{food_item}" in sentence:
                sentence = sentence.replace("{food_item}", random.choice(entities["food_item"]))
            if "{order_id}" in sentence:
                sentence = sentence.replace("{order_id}", random.choice(entities["order_id"]))
                
            examples_set.add(sentence)
            
        # Determine appropriate entity list for this intent
        intent_entities = []
        if intent_name == "book_flight":
            intent_entities = ["location", "date"]
        elif intent_name == "order_food":
            intent_entities = ["food_item"]
        elif intent_name == "check_weather":
            intent_entities = ["location", "date"]
        elif intent_name == "cancel_order":
            intent_entities = ["order_id", "food_item"]
        elif intent_name == "track_order":
            intent_entities = ["order_id", "food_item"]
        elif intent_name == "talk_to_human":
            intent_entities = []
            
        generated_intents.append({
            "name": intent_name,
            "examples": list(examples_set),
            "entities": intent_entities
        })

    final_dataset = {
        "intents": generated_intents,
        "entities": entities
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_dataset, f, indent=2)
        
    print(f"Successfully generated {target_examples_per_intent * len(templates)} examples in {output_path}")

if __name__ == "__main__":
    generate_large_dataset()
