use super::classifier::PsyAttentionClassifier;
use super::psychological_features::PsychologicalFeatureExtractor;
use csv::ReaderBuilder;
use rand::seq::SliceRandom;
use rand::thread_rng;
use serde::Deserialize;
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;

#[derive(Debug, Deserialize, Clone)]
struct MbtiRecord {
    #[serde(rename = "type")]
    mbti_type: String,
    posts: String,
}

fn calculate_accuracy(predictions: &[String], actual: &[String]) -> f64 {
    let correct = predictions
        .iter()
        .zip(actual.iter())
        .filter(|(pred, act)| pred == act)
        .count();
    correct as f64 / predictions.len() as f64
}

pub fn main_psyattention(_args: Vec<String>) -> Result<(), Box<dyn Error>> {
    println!("=== PsyAttention: MBTI Personality Classifier ===\n");
    println!("Simplified psychological feature extraction\n");

    // Load data
    println!("Loading data from CSV...");
    let file = File::open("data/mbti_1.csv")?;
    let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);

    let mut records: Vec<MbtiRecord> = Vec::new();
    for result in rdr.deserialize() {
        let record: MbtiRecord = result?;
        records.push(record);
    }

    println!("Loaded {} records\n", records.len());

    // Shuffle data
    let mut rng = thread_rng();
    records.shuffle(&mut rng);

    // Split train/test (80/20)
    let split_idx = (records.len() as f64 * 0.8) as usize;
    let train_records = &records[..split_idx];
    let test_records = &records[split_idx..];

    println!("Train size: {}", train_records.len());
    println!("Test size: {}\n", test_records.len());

    // Show feature extraction example
    println!("=== Psychological Feature Analysis ===");
    let extractor = PsychologicalFeatureExtractor::new();
    let sample_text = &train_records[0].posts;
    let sample_features = extractor.extract_features_named(sample_text);

    println!(
        "\nSample text: {}...",
        &sample_text.chars().take(100).collect::<String>()
    );
    println!("\nExtracted psychological features:");
    for (name, value) in sample_features.iter() {
        println!("  {}: {:.4}%", name, value);
    }
    println!();

    // Train classifier
    println!("=== Training PsyAttention Classifier ===\n");
    println!("Using Top 9 psychological features:");
    for (i, name) in PsychologicalFeatureExtractor::feature_names()
        .iter()
        .enumerate()
    {
        let weight = PsychologicalFeatureExtractor::feature_weights()[i];
        println!("  {}. {} - Weight: {:.2}", i + 1, name, weight);
    }
    println!();

    let mut classifier = PsyAttentionClassifier::new();

    let train_texts: Vec<String> = train_records.iter().map(|r| r.posts.clone()).collect();
    let train_labels: Vec<String> = train_records.iter().map(|r| r.mbti_type.clone()).collect();

    println!("Training with {} samples...", train_texts.len());
    classifier.train(&train_texts, &train_labels);
    println!("Training complete!\n");

    // Evaluate on training set
    println!("=== Evaluating on Training Set ===");
    let train_predictions: Vec<String> = train_texts
        .iter()
        .map(|text| classifier.predict(text))
        .collect();
    let train_accuracy = calculate_accuracy(&train_predictions, &train_labels);
    println!("Training accuracy: {:.2}%", train_accuracy * 100.0);

    // Evaluate on test set
    println!("\n=== Evaluating on Test Set ===");
    let test_texts: Vec<String> = test_records.iter().map(|r| r.posts.clone()).collect();
    let test_labels: Vec<String> = test_records.iter().map(|r| r.mbti_type.clone()).collect();

    let test_predictions: Vec<String> = test_texts
        .iter()
        .map(|text| classifier.predict(text))
        .collect();
    let test_accuracy = calculate_accuracy(&test_predictions, &test_labels);
    println!("Test accuracy: {:.2}%", test_accuracy * 100.0);

    // Show class distribution
    println!("\n=== Class Distribution in Training Set ===");
    let mut class_counts: HashMap<String, usize> = HashMap::new();
    for label in &train_labels {
        *class_counts.entry(label.clone()).or_insert(0) += 1;
    }
    let mut sorted_classes: Vec<_> = class_counts.iter().collect();
    sorted_classes.sort_by(|a, b| b.1.cmp(a.1));

    for (class, count) in sorted_classes.iter().take(10) {
        println!(
            "{}: {} ({:.1}%)",
            class,
            count,
            (**count as f64 / train_labels.len() as f64) * 100.0
        );
    }

    // Show detailed predictions with feature analysis
    println!("\n=== Sample Predictions with Feature Analysis ===");
    for (i, record) in test_records.iter().enumerate().take(3) {
        println!("\n--- Sample {} ---", i + 1);
        println!(
            "Text: {}...",
            &record.posts.chars().take(100).collect::<String>()
        );
        println!("Actual: {}", record.mbti_type);

        let prediction = classifier.predict(&record.posts);
        println!("Predicted: {}", prediction);
        println!("Match: {}", prediction == record.mbti_type);

        // Show feature analysis
        let features = classifier.analyze_features(&record.posts);
        println!("\nPsychological features detected:");
        let mut feature_vec: Vec<_> = features.iter().collect();
        feature_vec.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

        for (name, value) in feature_vec.iter().take(5) {
            if **value > 0.0 {
                println!("  {}: {:.2}%", name, value);
            }
        }

        // Show probability distribution
        let proba = classifier.predict_proba(&test_records[i].posts);
        let mut proba_vec: Vec<_> = proba.iter().collect();
        proba_vec.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

        println!("\nTop 3 predictions:");
        for (class, prob) in proba_vec.iter().take(3) {
            println!("  {}: {:.2}%", class, **prob * 100.0);
        }
    }

    println!("\n=== Training Complete ===");
    println!("\nCurrent accuracy: {:.2}%", test_accuracy * 100.0);
    println!("\nNote: Further improvements could include:");
    println!("  - All 930 psychological features (we use top 9)");
    println!("  - 8-layer Transformer encoder (we use weighted attention)");
    println!("  - BERT fine-tuning (not implemented)");
    println!("  - GPU training (we use CPU)");
    println!("\nOur simplified implementation demonstrates the core concepts!");

    Ok(())
}
