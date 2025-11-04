//! Basic usage example of psycial library

use psycial::{load_data, split_data, BaselineClassifier};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Basic Psycial Library Usage ===\n");

    // 1. Load data
    println!("Loading data...");
    let records = load_data("data/mbti_1.csv")?;
    println!("✓ Loaded {} records\n", records.len());

    // 2. Split into train/test
    println!("Splitting data (80/20)...");
    let (train_data, test_data) = split_data(&records, 0.8);
    println!("✓ Train: {} records", train_data.len());
    println!("✓ Test: {} records\n", test_data.len());

    // 3. Train a classifier
    println!("Training baseline classifier...");
    let mut classifier = BaselineClassifier::new();
    classifier.train(&train_data)?;
    println!("✓ Training complete\n");

    // 4. Make predictions
    println!("Making predictions...");
    let test_texts = vec![
        "I love spending time alone reading books and thinking deeply about life",
        "I enjoy being around people and organizing social events",
        "I prefer planning everything in advance and staying organized",
        "I like to keep my options open and be spontaneous",
    ];

    for text in test_texts {
        let prediction = classifier.predict(text)?;
        println!("Text: \"{}...\"", &text[..50.min(text.len())]);
        println!("Prediction: {}\n", prediction);
    }

    // 5. Save model
    println!("Saving model...");
    classifier.save("models/example_model.json")?;
    println!("✓ Model saved to models/example_model.json\n");

    // 6. Load model
    println!("Loading saved model...");
    let loaded_classifier = BaselineClassifier::load("models/example_model.json")?;
    println!("✓ Model loaded successfully\n");

    let prediction = loaded_classifier.predict("Testing with loaded model")?;
    println!("Loaded model prediction: {}", prediction);

    Ok(())
}
