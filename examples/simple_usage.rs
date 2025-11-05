/// Example: Using Psycial as a library
///
/// This demonstrates how to use the core MBTI classification functionality
/// in your own Rust application.

#[cfg(feature = "bert")]
use psycial::{load_data, MultiTaskGpuMLP, RustBertEncoder};
use std::error::Error;

#[cfg(feature = "bert")]
fn main() -> Result<(), Box<dyn Error>> {
    println!("========================================");
    println!("  Psycial Library Usage Example");
    println!("========================================\n");

    // 1. Load data
    println!("1. Loading MBTI data...");
    let records = load_data("data/mbti_1.csv")?;
    println!("   Loaded {} records\n", records.len());

    // 2. Initialize BERT encoder
    println!("2. Initializing BERT encoder...");
    let bert = RustBertEncoder::new()?;
    println!("   Embedding dimension: {}\n", bert.embedding_dim());

    // 3. Extract features from a small sample
    println!("3. Extracting features (first 100 samples)...");
    let sample_size = 100.min(records.len());
    let texts: Vec<String> = records[..sample_size]
        .iter()
        .map(|r| r.posts.clone())
        .collect();
    let labels: Vec<String> = records[..sample_size]
        .iter()
        .map(|r| r.mbti_type.clone())
        .collect();

    // Use batch extraction for better GPU utilization
    let features = bert.extract_features_batch(&texts)?;
    println!("   Extracted {} feature vectors\n", features.len());

    // 4. Create and train a small multi-task model
    println!("4. Creating multi-task model...");
    let mut model = MultiTaskGpuMLP::new(
        384,           // BERT embedding dimension
        vec![128, 64], // Smaller network for demo
        0.001,         // Learning rate
        0.3,           // Dropout
    );

    println!("5. Training model (5 epochs for demo)...");
    model.train(&features, &labels, 5, 16);

    // 6. Make predictions
    println!("\n6. Testing predictions...");
    let test_texts = vec![
        "I love organizing everything and making detailed plans.",
        "I prefer spontaneous adventures and going with the flow.",
        "I enjoy deep philosophical discussions about abstract concepts.",
    ];

    for text in test_texts {
        let feat = bert.extract_features(text)?;
        let prediction = model.predict(&feat);
        println!("   Text: \"{}...\"", &text[..50.min(text.len())]);
        println!("   Predicted: {}\n", prediction);
    }

    // 7. Save model
    println!("7. Saving model...");
    model.save("examples/demo_model.pt")?;
    println!("   Model saved to examples/demo_model.pt\n");

    println!("========================================");
    println!("  Example completed successfully!");
    println!("========================================\n");

    Ok(())
}

#[cfg(not(feature = "bert"))]
fn main() {
    eprintln!("This example requires the 'bert' feature.");
    eprintln!("Run with: cargo run --example simple_usage --features bert");
}
