//! PsyAttention: MBTI Classification with Psychological and Linguistic Features
//!
//! This module implements a comprehensive MBTI personality type classifier that combines:
//!
//! - **Psychological Features**: Emotion, sentiment, subjectivity, and other psychological indicators
//! - **Linguistic Features**: Style metrics from TAALES, TAACO, and SEANCE frameworks
//! - **BERT Embeddings**: Contextualized semantic representations
//! - **Attention Mechanism**: Weighted feature fusion for improved accuracy
//!
//! ## Feature Extraction Frameworks
//!
//! ### SEANCE (Sentiment and Emotion in Language Communication Evaluator)
//! - Emotion scores (anger, fear, joy, sadness, surprise)
//! - Sentiment polarity and intensity
//! - Subjectivity and objectivity measures
//!
//! ### TAALES (Tool for the Automatic Analysis of Lexical Sophistication)
//! - Lexical diversity and sophistication
//! - Word frequency and age-of-acquisition
//! - Academic and content word measures
//!
//! ### TAACO (Tool for the Automatic Analysis of Cohesion)
//! - Text cohesion metrics
//! - Referential and semantic overlap
//! - Connective and logical flow measures
//!
//! ## Architecture
//!
//! The PsyAttention model uses a multi-stage pipeline:
//!
//! 1. **Feature Extraction**: Extract 930 psychological and linguistic features
//! 2. **Feature Selection**: Reduce to 108 most informative features
//! 3. **BERT Encoding**: Generate 384-dimensional semantic embeddings
//! 4. **Attention Fusion**: Combine features with learned attention weights
//! 5. **Classification**: MLP classifier for final MBTI prediction
//!
//! ## Variants
//!
//! - **Simple**: 9 core psychological features (emotion + sentiment)
//! - **Full**: 930→108 selected features + BERT
//! - **BERT-only**: Pure BERT embeddings without psychological features
//!
//! ## Usage
//!
//! ```bash
//! # Train simple PsyAttention model
//! ./target/release/psycial psyattention
//!
//! # Train full PsyAttention with feature selection
//! ./target/release/psycial psyattention-full
//!
//! # BERT-integrated classifier
//! ./target/release/psycial psyattention-bert
//! ```

// Core modules used in production
pub mod bert_classifier;
pub mod bert_rustbert;
pub mod full_classifier;

// Supporting modules
pub mod attention_encoder;
pub mod classifier;
pub mod full_features;
pub mod fusion_layer;
pub mod psychological_features;
pub mod seance;
pub mod taaco;
pub mod taales;

// Main entry point for simple psyattention
pub mod simple;
pub use simple::main_psyattention;
