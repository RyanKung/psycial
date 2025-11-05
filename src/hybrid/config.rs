//! Configuration structures for the hybrid MBTI classifier.
//!
//! This module provides strongly-typed configuration management using TOML files.
//! The configuration covers data paths, feature extraction, model architecture,
//! training hyperparameters, and output paths.

use serde::Deserialize;
use std::error::Error;

/// Main configuration structure loaded from `config.toml`.
#[derive(Debug, Deserialize)]
pub struct Config {
    /// Data loading configuration
    pub data: DataConfig,
    /// Feature extraction configuration
    pub features: FeaturesConfig,
    /// Model architecture configuration
    pub model: ModelConfig,
    /// Training hyperparameters
    pub training: TrainingConfig,
    /// Output paths configuration
    pub output: OutputConfig,
}

/// Data loading configuration.
#[derive(Debug, Deserialize)]
pub struct DataConfig {
    /// Path to the CSV dataset file
    pub csv_path: String,
    /// Train/test split ratio (e.g., 0.8 = 80% train, 20% test)
    pub train_split: f64,
}

/// Feature extraction configuration.
#[derive(Debug, Deserialize)]
pub struct FeaturesConfig {
    /// Maximum number of TF-IDF features to extract
    pub max_tfidf_features: usize,
    /// Whether to use psychological features from SEANCE, TAACO, TAALES
    #[serde(default)]
    pub use_psychological_features: bool,
    /// Psychological feature type: "simple" (9), "selected" (108), "full" (930)
    #[serde(default = "default_psy_feature_type")]
    pub psy_feature_type: String,
}

fn default_psy_feature_type() -> String {
    "selected".to_string()
}

/// Model architecture configuration.
#[derive(Debug, Deserialize)]
pub struct ModelConfig {
    /// Model type: "single" (16-way classification) or "multitask" (4 binary classifiers)
    pub model_type: String,
    /// Hidden layer sizes (e.g., [1024, 512, 256])
    pub hidden_layers: Vec<i64>,
    /// Learning rate for Adam optimizer
    pub learning_rate: f64,
    /// Dropout rate for regularization
    pub dropout_rate: f64,
    /// Weight decay (L2 regularization) for Adam optimizer
    pub weight_decay: f64,
}

/// Training hyperparameters.
#[derive(Debug, Deserialize)]
pub struct TrainingConfig {
    /// Number of training epochs (for single-task model)
    pub epochs: i64,
    /// Batch size for training
    pub batch_size: i64,
    /// Batch size for BERT feature extraction
    #[allow(dead_code)]
    pub bert_batch_size: usize,
    /// Per-dimension epochs for multi-task model [E/I, S/N, T/F, J/P]
    /// Different dimensions have different data distributions:
    /// - E/I: 77% vs 23% (moderate imbalance)
    /// - S/N: 86% vs 14% (severe imbalance) ← needs more epochs
    /// - T/F: 46% vs 54% (most balanced) ← needs fewer epochs
    /// - J/P: 40% vs 60% (moderate imbalance)
    #[serde(default = "default_per_dimension_epochs")]
    pub per_dimension_epochs: Vec<i64>,
    /// Loss weights for each dimension [E/I, S/N, T/F, J/P]
    /// Higher weight = more importance in total loss
    /// Recommended: give more weight to weaker dimensions
    #[serde(default = "default_dimension_loss_weights")]
    pub dimension_loss_weights: Vec<f64>,
}

fn default_per_dimension_epochs() -> Vec<i64> {
    vec![25, 35, 20, 25] // [E/I, S/N, T/F, J/P]
}

fn default_dimension_loss_weights() -> Vec<f64> {
    vec![1.0, 1.0, 1.5, 1.2] // [E/I, S/N, T/F, J/P] - boost T/F since it's weakest
}

/// Output paths configuration.
#[derive(Debug, Deserialize)]
pub struct OutputConfig {
    /// Directory to save model files
    pub model_dir: String,
    /// TF-IDF vectorizer filename
    pub tfidf_file: String,
    /// MLP weights filename
    pub mlp_file: String,
    /// Class mapping filename (for single-task models)
    pub class_mapping_file: String,
}

impl Config {
    /// Load configuration from a TOML file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the configuration file
    ///
    /// # Returns
    ///
    /// Returns `Ok(Config)` if successful, or an error if the file cannot be read
    /// or parsed.
    pub fn load(path: &str) -> Result<Self, Box<dyn Error>> {
        let content = std::fs::read_to_string(path)?;
        let config: Config = toml::from_str(&content)?;
        Ok(config)
    }
}

impl Default for Config {
    /// Get default configuration if `config.toml` is not available.
    fn default() -> Self {
        Config {
            data: DataConfig {
                csv_path: "data/mbti_1.csv".to_string(),
                train_split: 0.8,
            },
            features: FeaturesConfig {
                max_tfidf_features: 5000,
                use_psychological_features: false,
                psy_feature_type: "selected".to_string(),
            },
            model: ModelConfig {
                model_type: "multitask".to_string(),
                hidden_layers: vec![1024, 512, 256],
                learning_rate: 0.001,
                dropout_rate: 0.5,
                weight_decay: 0.0,
            },
            training: TrainingConfig {
                epochs: 25,
                batch_size: 64,
                bert_batch_size: 64,
                per_dimension_epochs: vec![25, 35, 20, 25],
                dimension_loss_weights: vec![1.0, 1.0, 1.5, 1.2],
            },
            output: OutputConfig {
                model_dir: "models".to_string(),
                tfidf_file: "tfidf_vectorizer.json".to_string(),
                mlp_file: "mlp_weights.pt".to_string(),
                class_mapping_file: "class_mapping.json".to_string(),
            },
        }
    }
}
