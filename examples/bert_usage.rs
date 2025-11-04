//! BERT-based classifier usage example
//!
//! Run with: cargo run --example bert_usage --features bert

#[cfg(feature = "bert")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use psycial::{load_data, split_data, BertClassifier};

    println!("=== BERT Classifier Usage ===\n");

    // Load data
    println!("Loading data...");
    let records = load_data("data/mbti_1.csv")?;
    println!("✓ Loaded {} records\n", records.len());

    // Split data
    let (train_data, test_data) = split_data(&records, 0.8);
    println!("Train: {}, Test: {}\n", train_data.len(), test_data.len());

    // Initialize BERT classifier
    println!("Initializing BERT classifier...");
    let mut classifier = BertClassifier::new()?;
    println!("✓ BERT initialized\n");

    // Train
    println!("Training BERT classifier...");
    classifier.train(&train_data)?;
    println!("✓ Training complete\n");

    // Predict
    let text = "I love coding and solving complex problems through logical analysis";
    let prediction = classifier.predict(text)?;
    println!("Text: {}", text);
    println!("Prediction: {}\n", prediction);

    println!("✓ Done!");

    Ok(())
}

#[cfg(not(feature = "bert"))]
fn main() {
    eprintln!("This example requires the 'bert' feature.");
    eprintln!("Run with: cargo run --example bert_usage --features bert");
    std::process::exit(1);
}
