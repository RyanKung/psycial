use crate::neural_net_gpu::GpuMLP;
use crate::neural_net_gpu_multitask::MultiTaskGpuMLP;
use crate::psyattention::bert_rustbert::RustBertEncoder;

use csv::ReaderBuilder;
use rand::seq::SliceRandom;
use rand::thread_rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::time::Instant;

// Configuration structures
#[derive(Debug, Deserialize)]
struct Config {
    data: DataConfig,
    features: FeaturesConfig,
    model: ModelConfig,
    training: TrainingConfig,
    output: OutputConfig,
}

#[derive(Debug, Deserialize)]
struct DataConfig {
    csv_path: String,
    train_split: f64,
}

#[derive(Debug, Deserialize)]
struct FeaturesConfig {
    max_tfidf_features: usize,
}

#[derive(Debug, Deserialize)]
struct ModelConfig {
    model_type: String, // "single" or "multitask"
    hidden_layers: Vec<i64>,
    learning_rate: f64,
    dropout_rate: f64,
}

#[derive(Debug, Deserialize)]
struct TrainingConfig {
    epochs: i64,
    batch_size: i64,
    #[allow(dead_code)]
    bert_batch_size: usize,
}

#[derive(Debug, Deserialize)]
struct OutputConfig {
    model_dir: String,
    tfidf_file: String,
    mlp_file: String,
    class_mapping_file: String,
}

impl Config {
    fn load(path: &str) -> Result<Self, Box<dyn Error>> {
        let content = std::fs::read_to_string(path)?;
        let config: Config = toml::from_str(&content)?;
        Ok(config)
    }
}

#[derive(Debug, Deserialize, Clone)]
struct MbtiRecord {
    #[serde(rename = "type")]
    mbti_type: String,
    posts: String,
}

// Simple TF-IDF implementation
#[derive(Serialize, Deserialize)]
struct TfidfVectorizer {
    vocabulary: HashMap<String, usize>,
    idf: Vec<f64>,
    max_features: usize,
}

impl TfidfVectorizer {
    fn new(max_features: usize) -> Self {
        TfidfVectorizer {
            vocabulary: HashMap::new(),
            idf: Vec::new(),
            max_features,
        }
    }

    fn fit(&mut self, documents: &[String]) {
        let mut word_doc_count: HashMap<String, usize> = HashMap::new();
        let mut word_freq: HashMap<String, usize> = HashMap::new();

        for doc in documents {
            let words: Vec<String> = doc
                .to_lowercase()
                .split_whitespace()
                .filter(|w| w.len() > 2)
                .map(|s| s.to_string())
                .collect();

            let unique_words: std::collections::HashSet<_> = words.iter().collect();
            for word in unique_words {
                *word_doc_count.entry(word.clone()).or_insert(0) += 1;
                *word_freq.entry(word.clone()).or_insert(0) += 1;
            }
        }

        // Select top max_features by frequency
        let mut word_freq_vec: Vec<_> = word_freq.iter().collect();
        word_freq_vec.sort_by(|a, b| b.1.cmp(a.1));

        for (idx, (word, _)) in word_freq_vec.iter().take(self.max_features).enumerate() {
            self.vocabulary.insert((*word).clone(), idx);
        }

        // Calculate IDF
        self.idf = vec![0.0; self.vocabulary.len()];
        let n_docs = documents.len() as f64;

        for (word, &idx) in &self.vocabulary {
            let doc_freq = *word_doc_count.get(word).unwrap_or(&1) as f64;
            self.idf[idx] = (n_docs / doc_freq).ln();
        }
    }

    fn transform(&self, document: &str) -> Vec<f64> {
        let mut tf = vec![0.0; self.vocabulary.len()];

        let words: Vec<String> = document
            .to_lowercase()
            .split_whitespace()
            .filter(|w| w.len() > 2)
            .map(|s| s.to_string())
            .collect();

        for word in words {
            if let Some(&idx) = self.vocabulary.get(&word) {
                tf[idx] += 1.0;
            }
        }

        // Normalize TF
        let total: f64 = tf.iter().sum();
        if total > 0.0 {
            for val in &mut tf {
                *val /= total;
            }
        }

        // Apply IDF and L2 normalize
        let mut tfidf: Vec<f64> = tf
            .iter()
            .zip(self.idf.iter())
            .map(|(&t, &i)| t * i)
            .collect();

        let norm = tfidf.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            for val in &mut tfidf {
                *val /= norm;
            }
        }

        tfidf
    }

    fn save(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self).map_err(std::io::Error::other)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    fn load(path: &str) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let vectorizer = serde_json::from_str(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Ok(vectorizer)
    }
}

// Note: GPU model saving disabled - GpuMLP contains non-serializable GPU tensors
// Future: implement custom save/load using tch::nn::VarStore

fn print_usage() {
    println!("Usage:");
    println!("  cargo run --release --features bert --bin psycial -- hybrid [COMMAND] [OPTIONS]\n");
    println!("Commands:");
    println!("  train              Train new GPU model (saves to models/)");
    println!("  predict TEXT       Predict single text (requires trained model)");
    println!("  help               Show this help\n");
    println!("Options:");
    println!(
        "  --multi-task       Use multi-task model (4 binary classifiers: E/I, S/N, T/F, J/P)"
    );
    println!("  --single-task      Use single-task model (16-way classification)");
    println!("                     Default: uses config.toml setting\n");
    println!("Examples:");
    println!("  ./target/release/psycial hybrid train --multi-task");
    println!("  ./target/release/psycial hybrid train --single-task");
    println!("  ./target/release/psycial hybrid predict \"I love solving problems\"");
}

pub fn main_hybrid(args: Vec<String>) -> Result<(), Box<dyn Error>> {
    let command = if args.len() > 1 {
        args[1].as_str()
    } else {
        "train"
    };

    match command {
        "train" => {
            // Check for model type flags
            let model_type_override = if args.contains(&"--multi-task".to_string()) {
                Some("multitask")
            } else if args.contains(&"--single-task".to_string()) {
                Some("single")
            } else {
                None
            };
            train_model(model_type_override)
        }
        "predict" => {
            if args.len() < 3 {
                println!("Error: TEXT argument required\n");
                print_usage();
                return Ok(());
            }
            predict_single(&args[2])
        }
        "help" | "--help" | "-h" => {
            print_usage();
            Ok(())
        }
        _ => {
            println!("Unknown command: {}\n", command);
            print_usage();
            Ok(())
        }
    }
}

fn train_model(model_type_override: Option<&str>) -> Result<(), Box<dyn Error>> {
    // Load configuration
    let mut config = Config::load("config.toml").unwrap_or_else(|e| {
        eprintln!("Warning: Could not load config.toml: {}", e);
        eprintln!("Using default configuration\n");
        Config {
            data: DataConfig {
                csv_path: "data/mbti_1.csv".to_string(),
                train_split: 0.8,
            },
            features: FeaturesConfig {
                max_tfidf_features: 5000,
            },
            model: ModelConfig {
                model_type: "multitask".to_string(),
                hidden_layers: vec![1024, 512, 256],
                learning_rate: 0.001,
                dropout_rate: 0.5,
            },
            training: TrainingConfig {
                epochs: 25,
                batch_size: 64,
                bert_batch_size: 64,
            },
            output: OutputConfig {
                model_dir: "models".to_string(),
                tfidf_file: "tfidf_vectorizer.json".to_string(),
                mlp_file: "mlp_weights.pt".to_string(),
                class_mapping_file: "class_mapping.json".to_string(),
            },
        }
    });

    // Apply command-line override if provided
    if let Some(model_type) = model_type_override {
        config.model.model_type = model_type.to_string();
        println!(
            "ℹ️  Model type overridden by command-line flag: {}\n",
            model_type
        );
    }

    println!("\n===================================================================");
    println!("  MBTI Classifier: GPU-Accelerated Hybrid Model");
    println!("  TF-IDF + BERT + GPU MLP");
    println!("===================================================================\n");

    println!("Configuration:");
    println!("  Data: {}", config.data.csv_path);
    println!(
        "  Train/Test split: {:.0}%/{:.0}%",
        config.data.train_split * 100.0,
        (1.0 - config.data.train_split) * 100.0
    );
    println!("  TF-IDF features: {}", config.features.max_tfidf_features);
    println!(
        "  Model type: {} {}",
        config.model.model_type,
        if config.model.model_type == "multitask" {
            "(4 binary classifiers: E/I, S/N, T/F, J/P)"
        } else {
            "(16-way classification)"
        }
    );
    let output_desc = if config.model.model_type == "multitask" {
        "4×2"
    } else {
        "16"
    };
    println!(
        "  Architecture: {} -> {:?} -> {}",
        config.features.max_tfidf_features + 384,
        config.model.hidden_layers,
        output_desc
    );
    println!("  Learning rate: {}", config.model.learning_rate);
    println!("  Dropout: {}", config.model.dropout_rate);
    println!("  Epochs: {}", config.training.epochs);
    println!("  Batch size: {}\n", config.training.batch_size);
    println!("===================================================================\n");

    // Load data
    println!("Loading dataset...");
    let start = Instant::now();
    let file = File::open(&config.data.csv_path)?;
    let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);
    let mut records: Vec<MbtiRecord> = rdr.deserialize().collect::<Result<_, _>>()?;
    println!(
        "  Loaded {} records ({:.2}s)\n",
        records.len(),
        start.elapsed().as_secs_f64()
    );

    // Shuffle and split
    let mut rng = thread_rng();
    records.shuffle(&mut rng);
    let split = (records.len() as f64 * config.data.train_split) as usize;
    let train_records = &records[..split];
    let test_records = &records[split..];

    println!(
        "Train: {} | Test: {}\n",
        train_records.len(),
        test_records.len()
    );
    println!("===================================================================\n");

    // Extract TF-IDF features
    println!("Building TF-IDF vectorizer (top 5000 words)...");
    let train_texts: Vec<String> = train_records.iter().map(|r| r.posts.clone()).collect();
    let train_labels: Vec<String> = train_records.iter().map(|r| r.mbti_type.clone()).collect();

    let mut tfidf = TfidfVectorizer::new(5000);
    tfidf.fit(&train_texts);
    println!("  Vocabulary size: {}\n", tfidf.vocabulary.len());

    // Initialize BERT
    println!("Initializing BERT encoder...");
    let bert_encoder = RustBertEncoder::new()?;

    println!("===================================================================\n");
    println!("Extracting hybrid features (TF-IDF + BERT)...");

    let train_start = Instant::now();

    // Extract TF-IDF features first
    println!("  Extracting TF-IDF features...");
    let tfidf_features: Vec<Vec<f64>> = train_texts
        .iter()
        .enumerate()
        .map(|(i, text)| {
            if (i + 1) % 1000 == 0 {
                println!("    TF-IDF: {}/{}", i + 1, train_texts.len());
            }
            tfidf.transform(text)
        })
        .collect();

    // Batch extract BERT features (GPU optimized!)
    println!("  Extracting BERT features (GPU batch mode)...");
    let batch_size = 64; // Process 64 texts at once on GPU
    let mut all_bert_features = Vec::new();

    for (batch_idx, chunk) in train_texts.chunks(batch_size).enumerate() {
        let chunk_vec: Vec<String> = chunk.to_vec();
        let bert_batch = bert_encoder.extract_features_batch(&chunk_vec)?;
        all_bert_features.extend(bert_batch);

        if (batch_idx + 1) % 10 == 0 {
            println!(
                "    BERT batch: {}/{}",
                (batch_idx + 1) * batch_size,
                train_texts.len()
            );
        }
    }

    // Combine TF-IDF + BERT features
    println!("  Combining features...");
    let mut train_features = Vec::new();
    for (tfidf_feat, bert_feat) in tfidf_features
        .into_iter()
        .zip(all_bert_features.into_iter())
    {
        let mut combined = tfidf_feat;
        combined.extend(bert_feat);
        train_features.push(combined);
    }

    println!(
        "\nTraining GPU-Accelerated MLP ({})...",
        config.model.model_type
    );
    let input_dim = config.features.max_tfidf_features + 384;
    println!(
        "  Input: {} features ({} TF-IDF + 384 BERT)",
        input_dim, config.features.max_tfidf_features
    );
    println!(
        "  Architecture: {} -> {:?} -> {}",
        input_dim, config.model.hidden_layers, output_desc
    );
    println!("  Optimizer: Adam");
    println!("  Learning rate: {}", config.model.learning_rate);
    println!("  Dropout: {}", config.model.dropout_rate);
    println!("  Epochs: {}\n", config.training.epochs);

    // Choose model based on configuration
    if config.model.model_type == "multitask" {
        let mut mlp = MultiTaskGpuMLP::new(
            input_dim as i64,
            config.model.hidden_layers.clone(),
            config.model.learning_rate,
            config.model.dropout_rate,
        );

        mlp.train(
            &train_features,
            &train_labels,
            config.training.epochs,
            config.training.batch_size,
        );

        // Continue with multitask evaluation below
        let eval_ctx = EvaluationContext {
            train_features: &train_features,
            train_labels: &train_labels,
            test_records,
            tfidf: &tfidf,
            bert_encoder: &bert_encoder,
            train_start,
            config: &config,
        };
        evaluate_multitask_model(&mlp, &eval_ctx)?;
    } else {
        let mut mlp = GpuMLP::new(
            input_dim as i64,
            config.model.hidden_layers.clone(),
            16, // 16 MBTI types
            config.model.learning_rate,
            config.model.dropout_rate,
        );

        mlp.train(
            &train_features,
            &train_labels,
            config.training.epochs,
            config.training.batch_size,
        );

        // Continue with single-task evaluation below
        let eval_ctx = EvaluationContext {
            train_features: &train_features,
            train_labels: &train_labels,
            test_records,
            tfidf: &tfidf,
            bert_encoder: &bert_encoder,
            train_start,
            config: &config,
        };
        evaluate_singletask_model(&mlp, &eval_ctx)?;
    }

    Ok(())
}

// Evaluation context to avoid too many arguments
struct EvaluationContext<'a> {
    train_features: &'a [Vec<f64>],
    train_labels: &'a [String],
    test_records: &'a [MbtiRecord],
    tfidf: &'a TfidfVectorizer,
    bert_encoder: &'a RustBertEncoder,
    train_start: Instant,
    config: &'a Config,
}

// Evaluation function for multi-task model
fn evaluate_multitask_model(
    mlp: &MultiTaskGpuMLP,
    ctx: &EvaluationContext,
) -> Result<(), Box<dyn Error>> {
    println!(
        "\nTotal training time: {:.2}s\n",
        ctx.train_start.elapsed().as_secs_f64()
    );
    println!("===================================================================\n");

    println!("Evaluation\n");

    // Training Set Evaluation
    println!("Training Set:");
    let train_predictions = mlp.predict_batch(ctx.train_features);
    let mut correct = 0;
    for (pred, label) in train_predictions.iter().zip(ctx.train_labels.iter()) {
        if pred == label {
            correct += 1;
        }
    }
    let train_acc = correct as f64 / ctx.train_labels.len() as f64;
    println!("  Accuracy: {:.2}%\n", train_acc * 100.0);

    // Test Set Evaluation
    println!("Test Set:");
    let test_start = Instant::now();
    let test_texts: Vec<String> = ctx.test_records.iter().map(|r| r.posts.clone()).collect();
    let test_labels: Vec<String> = ctx
        .test_records
        .iter()
        .map(|r| r.mbti_type.clone())
        .collect();

    let mut test_features = Vec::new();
    for (i, text) in test_texts.iter().enumerate() {
        if (i + 1) % 500 == 0 {
            println!("  Test: {}/{}", i + 1, test_texts.len());
        }
        let tfidf_feat = ctx.tfidf.transform(text);
        let bert_feat = ctx.bert_encoder.extract_features(text)?;
        let mut combined = tfidf_feat;
        combined.extend(bert_feat);
        test_features.push(combined);
    }

    let test_predictions = mlp.predict_batch(&test_features);
    let mut correct = 0;
    for (pred, label) in test_predictions.iter().zip(test_labels.iter()) {
        if pred == label {
            correct += 1;
        }
    }
    let test_acc = correct as f64 / test_labels.len() as f64;
    println!("  Accuracy: {:.2}%", test_acc * 100.0);
    println!("  Time: {:.2}s\n", test_start.elapsed().as_secs_f64());

    print_results(test_acc);

    // Save model
    save_model(mlp, ctx.tfidf, &ctx.config.output)?;

    Ok(())
}

// Evaluation function for single-task model
fn evaluate_singletask_model(mlp: &GpuMLP, ctx: &EvaluationContext) -> Result<(), Box<dyn Error>> {
    println!(
        "\nTotal training time: {:.2}s\n",
        ctx.train_start.elapsed().as_secs_f64()
    );
    println!("===================================================================\n");

    println!("Evaluation\n");

    // Training Set Evaluation
    println!("Training Set:");
    let train_predictions = mlp.predict_batch(ctx.train_features);
    let mut correct = 0;
    for (pred, label) in train_predictions.iter().zip(ctx.train_labels.iter()) {
        if pred == label {
            correct += 1;
        }
    }
    let train_acc = correct as f64 / ctx.train_labels.len() as f64;
    println!("  Accuracy: {:.2}%\n", train_acc * 100.0);

    // Test Set Evaluation
    println!("Test Set:");
    let test_start = Instant::now();
    let test_texts: Vec<String> = ctx.test_records.iter().map(|r| r.posts.clone()).collect();
    let test_labels: Vec<String> = ctx
        .test_records
        .iter()
        .map(|r| r.mbti_type.clone())
        .collect();

    let mut test_features = Vec::new();
    for (i, text) in test_texts.iter().enumerate() {
        if (i + 1) % 500 == 0 {
            println!("  Test: {}/{}", i + 1, test_texts.len());
        }
        let tfidf_feat = ctx.tfidf.transform(text);
        let bert_feat = ctx.bert_encoder.extract_features(text)?;
        let mut combined = tfidf_feat;
        combined.extend(bert_feat);
        test_features.push(combined);
    }

    let test_predictions = mlp.predict_batch(&test_features);
    let mut correct = 0;
    for (pred, label) in test_predictions.iter().zip(test_labels.iter()) {
        if pred == label {
            correct += 1;
        }
    }
    let test_acc = correct as f64 / test_labels.len() as f64;
    println!("  Accuracy: {:.2}%", test_acc * 100.0);
    println!("  Time: {:.2}s\n", test_start.elapsed().as_secs_f64());

    print_results(test_acc);

    // Save model
    save_model_singletask(mlp, ctx.tfidf, &ctx.config.output)?;

    Ok(())
}

// Helper function to print results
fn print_results(test_acc: f64) {
    println!("===================================================================\n");
    println!("Final Results\n");
    println!("+--------------------------------------------+----------+");
    println!("| Method                                     | Accuracy |");
    println!("+--------------------------------------------+----------+");
    println!("| Baseline (TF-IDF + Naive Bayes)            |  21.73%  |");
    println!("| BERT + MLP (single-task)                   |  31.99%  |");
    println!("| Hybrid (single-task, previous)             |  49.16%  |");
    println!(
        "| Hybrid (current)                           | {:>6.2}%  |",
        test_acc * 100.0
    );
    println!("| Paper Target (BERT + Transformer)          |  86.30%  |");
    println!("+--------------------------------------------+----------+\n");

    let improvement = (test_acc - 0.2173) / 0.2173 * 100.0;
    let vs_paper = test_acc / 0.8630 * 100.0;

    println!("Analysis:");
    println!("  Improvement over baseline: {:.1}%", improvement);
    println!("  Progress toward paper: {:.1}%", vs_paper);
    println!("  vs Random: {:.1}x better\n", test_acc / 0.0625);

    if test_acc > 0.55 {
        println!("🎉 OUTSTANDING: Multi-task approach showing excellent results!");
    } else if test_acc > 0.48 {
        println!("✅ EXCELLENT: Significant improvement!");
    } else if test_acc > 0.35 {
        println!("👍 GOOD: Hybrid approach working well");
    } else if test_acc > 0.30 {
        println!("📊 SUCCESS: Above BERT-only baseline");
    } else {
        println!("📝 NOTE: Results at expected level");
    }

    println!("\n===================================================================\n");
}

// Save multi-task model
fn save_model(
    mlp: &MultiTaskGpuMLP,
    tfidf: &TfidfVectorizer,
    output: &OutputConfig,
) -> Result<(), Box<dyn Error>> {
    std::fs::create_dir_all(&output.model_dir)?;

    println!("Saving model...");

    // Add suffix for multi-task models
    let tfidf_file = output.tfidf_file.replace(".json", "_multitask.json");
    let mlp_file = output.mlp_file.replace(".pt", "_multitask.pt");

    let tfidf_path = format!("{}/{}", output.model_dir, tfidf_file);
    let mlp_path = format!("{}/{}", output.model_dir, mlp_file);

    tfidf.save(&tfidf_path)?;
    mlp.save(&mlp_path)?;

    println!("\n✓ Multi-Task Model saved:");
    println!("  - {}", tfidf_path);
    println!("  - {}", mlp_path);

    println!("\n===================================================================\n");
    println!("Training complete!\n");
    println!("Note: Multi-task model uses different files to avoid conflicts.");
    println!("To predict: ./target/release/psycial hybrid predict \"your text here\"\n");

    Ok(())
}

// Save single-task model
fn save_model_singletask(
    mlp: &GpuMLP,
    tfidf: &TfidfVectorizer,
    output: &OutputConfig,
) -> Result<(), Box<dyn Error>> {
    std::fs::create_dir_all(&output.model_dir)?;

    println!("Saving model...");

    // Add suffix for single-task models
    let tfidf_file = output.tfidf_file.replace(".json", "_single.json");
    let mlp_file = output.mlp_file.replace(".pt", "_single.pt");
    let class_file = output.class_mapping_file.replace(".json", "_single.json");

    let tfidf_path = format!("{}/{}", output.model_dir, tfidf_file);
    let mlp_path = format!("{}/{}", output.model_dir, mlp_file);
    let class_path = format!("{}/{}", output.model_dir, class_file);

    tfidf.save(&tfidf_path)?;
    mlp.save(&mlp_path)?;
    mlp.save_class_mapping(&class_path)?;

    println!("\n✓ Single-Task Model saved:");
    println!("  - {}", tfidf_path);
    println!("  - {}", mlp_path);
    println!("  - {}", class_path);

    println!("\n===================================================================\n");
    println!("Training complete!\n");
    println!("Note: Single-task model uses different files to avoid conflicts.");
    println!("To predict: ./target/release/psycial hybrid predict \"your text here\"\n");

    Ok(())
}

fn predict_single(text: &str) -> Result<(), Box<dyn Error>> {
    println!("\n===================================================================");
    println!("  MBTI Classifier: GPU Prediction");
    println!("===================================================================\n");

    println!("Loading model...");

    // Load TF-IDF vectorizer
    let tfidf = TfidfVectorizer::load("models/tfidf_vectorizer.json")?;
    println!("  ✓ TF-IDF loaded");

    // Load MLP
    let mut mlp = GpuMLP::load("models/mlp_weights.pt", 5384, vec![1024, 512, 256], 16, 0.4)?;
    mlp.load_class_mapping("models/class_mapping.json")?;
    println!("  ✓ MLP loaded\n");

    // Initialize BERT
    println!("Initializing BERT...");
    let bert_encoder = RustBertEncoder::new()?;

    println!("\nInput text:");
    let display = if text.len() > 100 {
        &format!("{}...", &text[..100])
    } else {
        text
    };
    println!("  {}\n", display);

    println!("Processing...");
    let start = Instant::now();

    // Extract features
    let tfidf_feat = tfidf.transform(text);
    let bert_feat = bert_encoder.extract_features(text)?;
    let mut combined = tfidf_feat;
    combined.extend(bert_feat);

    // Predict
    let prediction = mlp.predict(&combined);

    println!("\n===================================================================");
    println!("  MBTI Type: {}", prediction);
    println!("  Time: {:.3}s", start.elapsed().as_secs_f64());
    println!("===================================================================\n");

    Ok(())
}
