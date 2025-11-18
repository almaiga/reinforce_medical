# Requirements Document

## Introduction

This feature refactors the data loading and preparation functionality for the medical self-play RL training system. The current implementation in `scripts/create_rl_training_data.py` uses a complex approach that creates 2x data points from each error case. The goal is to simplify this to use the cleaner approach from `scripts/data.py`, which directly assigns game categories to individual samples in a 4-way balanced split (25% each category), without creating duplicate data points or few-shot examples.

## Glossary

- **MEDEC Dataset**: Medical Error Detection and Correction dataset containing medical notes with errors and their corrected versions
- **Game Category**: One of four categories in the self-play structure: vanilla_harmful, adversarial_harmful, vanilla_benign, adversarial_benign
- **Error Note**: A medical note containing one or more errors
- **Clean Note**: The corrected version of an error note
- **4-Way Split**: Balanced distribution where each game category receives exactly 25% of samples
- **RL Training Data**: Training data formatted for reinforcement learning with the OpenRLHF framework
- **Data Loading Module**: The system component responsible for loading and preparing MEDEC data

## Requirements

### Requirement 1

**User Story:** As a developer, I want a simplified data loading approach, so that the code is easier to understand and maintain.

#### Acceptance Criteria

1. WHEN the Data Loading Module loads MEDEC data THEN the system SHALL filter to only rows where Error Flag equals 1
2. WHEN the Data Loading Module processes samples THEN the system SHALL filter out rows with empty Text or Corrected Text fields
3. WHEN the Data Loading Module creates the dataset THEN the system SHALL use a single-pass approach that assigns one game category per sample
4. WHEN the Data Loading Module creates the dataset THEN the system SHALL NOT create duplicate data points from the same source row
5. WHEN the Data Loading Module shuffles data THEN the system SHALL use a fixed random seed for reproducibility

### Requirement 2

**User Story:** As a data scientist, I want a balanced 4-way game structure, so that the RL training has equal representation across all game categories.

#### Acceptance Criteria

1. WHEN the Data Loading Module creates the 4-way split THEN the system SHALL assign exactly 25% of samples to vanilla_harmful category
2. WHEN the Data Loading Module creates the 4-way split THEN the system SHALL assign exactly 25% of samples to adversarial_harmful category
3. WHEN the Data Loading Module creates the 4-way split THEN the system SHALL assign exactly 25% of samples to vanilla_benign category
4. WHEN the Data Loading Module creates the 4-way split THEN the system SHALL assign exactly 25% of samples to adversarial_benign category
5. WHEN the total number of samples is not divisible by 4 THEN the system SHALL use floor division to determine quarter samples

### Requirement 3

**User Story:** As a developer, I want proper data extraction and formatting, so that the RL training receives correctly structured input.

#### Acceptance Criteria

1. WHEN the Data Loading Module extracts error notes THEN the system SHALL use the Text column and strip whitespace
2. WHEN the Data Loading Module extracts clean notes THEN the system SHALL use the Corrected Text column and strip whitespace
3. WHEN the Data Loading Module extracts error types THEN the system SHALL use the Error Type column for harmful categories
4. WHEN the Data Loading Module processes benign categories THEN the system SHALL set error_type to "none" for vanilla_benign samples
5. WHEN the Data Loading Module creates output records THEN the system SHALL include game_category, error_note, clean_note, and error_type fields

### Requirement 4

**User Story:** As a developer, I want clear logging and verification, so that I can confirm the data loading process completed correctly.

#### Acceptance Criteria

1. WHEN the Data Loading Module loads data THEN the system SHALL log the number of available rows with errors
2. WHEN the Data Loading Module creates the 4-way split THEN the system SHALL log the count for each game category
3. WHEN the Data Loading Module creates the 4-way split THEN the system SHALL log the percentage distribution for each game category
4. WHEN the Data Loading Module completes THEN the system SHALL log the total number of output records
5. WHEN the Data Loading Module saves output THEN the system SHALL log the output file path

### Requirement 5

**User Story:** As a developer, I want to remove few-shot example generation, so that the code focuses only on RL training data preparation.

#### Acceptance Criteria

1. WHEN the Data Loading Module executes THEN the system SHALL NOT create few-shot example datasets
2. WHEN the Data Loading Module returns data THEN the system SHALL return only the training dataset
3. WHEN the Data Loading Module processes samples THEN the system SHALL NOT reserve samples for few-shot examples
4. WHEN the Data Loading Module completes THEN the system SHALL use all available samples for the 4-way split

### Requirement 6

**User Story:** As a developer, I want the refactored code to maintain compatibility with existing interfaces, so that downstream components continue to work without modification.

#### Acceptance Criteria

1. WHEN the Data Loading Module saves output THEN the system SHALL use JSONL format with one record per line
2. WHEN the Data Loading Module creates records THEN the system SHALL maintain the same field names as the current implementation
3. WHEN the Data Loading Module is invoked THEN the system SHALL accept the same command-line arguments as the current script
4. WHEN the Data Loading Module saves files THEN the system SHALL use the same output directory structure as the current implementation
5. WHEN the Data Loading Module completes THEN the system SHALL return the same exit codes as the current implementation
