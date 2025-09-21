# 03A - Parquet Loader Implementation

## Overview

This document provides a development checklist and acceptance criteria for implementing the parquet data loader based on the design in `03-load-parquet.md`.

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1-2)

#### 1.1 Project Structure
- [ ] Create `src/parquet_loader/` directory
- [ ] Create `src/parquet_loader/__init__.py`
- [ ] Create `src/parquet_loader/exceptions.py`
- [ ] Create `src/parquet_loader/config.py`
- [ ] Create `src/parquet_loader/file_discovery.py`
- [ ] Create `src/parquet_loader/idempotency.py`
- [ ] Create `src/parquet_loader/audit.py`
- [ ] Create `src/parquet_loader/loader.py`

#### 1.2 Configuration System
- [ ] Implement `ParquetLoaderConfig` Pydantic model
- [ ] Add configuration validation (fail-fast)
- [ ] Create `config/parquet_loader_config.yaml` template
- [ ] Add configuration loading with error handling

**Acceptance Criteria:**
- Configuration file must be explicitly provided (no defaults)
- Missing required fields cause immediate failure with clear error messages
- Invalid root directory path causes immediate failure
- Configuration validation covers all required fields

#### 1.3 Exception Hierarchy
- [ ] Implement `ParquetLoaderError` base class
- [ ] Implement `ConfigError` for configuration issues
- [ ] Implement `DataNotFoundError` for missing data
- [ ] Implement `InvalidFilenameError` for pattern mismatches
- [ ] Implement `DataQualityError` for data validation issues
- [ ] Implement `IncompleteContextError` for prediction context issues
- [ ] Implement `SchemaValidationError` for schema issues

**Acceptance Criteria:**
- All exceptions inherit from `ParquetLoaderError`
- Exception messages are clear and actionable
- Exceptions include relevant context (file paths, configuration values)

### Phase 2: File Discovery (Week 2-3)

#### 2.1 FileDiscovery Class
- [ ] Implement `FileDiscovery` class
- [ ] Add filename pattern parsing with regex
- [ ] Implement range-based file discovery
- [ ] Add `ParquetFileInfo` dataclass
- [ ] Handle missing directories gracefully

**Acceptance Criteria:**
- Discovers files matching pattern: `{SYMBOL}_{TIMEFRAME}_{HORIZON}_{YEAR}_{MONTH}_{HASH}.{EXT}`
- Supports year ranges: `year_range=(2014, 2016)`
- Supports month ranges: `month_range=(1, 6)`
- Skips missing year/month directories without failing
- Returns sorted list of `ParquetFileInfo` objects
- Validates filename patterns and raises `InvalidFilenameError` for mismatches

#### 2.2 Range Specification
- [ ] Support single year: `year=2014`
- [ ] Support year range: `year_range=(2014, 2016)`
- [ ] Support single month: `month=3`
- [ ] Support month range: `month_range=(1, 6)`
- [ ] Support all months: `month=None`
- [ ] Validate range parameters

**Acceptance Criteria:**
- Either `year` or `year_range` must be specified
- Year ranges are inclusive: `(2014, 2016)` includes both 2014 and 2016
- Month ranges are inclusive: `(1, 6)` includes months 1-6
- Missing directories are skipped, not treated as errors
- Returns empty list if no files found (no error)

### Phase 3: Idempotency System (Week 3-4)

#### 3.1 IdempotencyTracker Class
- [ ] Implement `IdempotencyTracker` class
- [ ] Add JSON-based state file management
- [ ] Implement file checksum calculation (MD5)
- [ ] Add `is_processed()` method
- [ ] Add `mark_processed()` method
- [ ] Handle state file creation and updates

**Acceptance Criteria:**
- State file: `.processed_files.json` in root directory
- File keys: `{SYMBOL}_{YEAR}_{MONTH:02d}_{HASH}`
- Tracks: file path, processed timestamp, checksum, status
- `is_processed()` returns `True` for already processed files
- `mark_processed()` updates state file atomically
- Checksum detects file changes and allows reprocessing

#### 3.2 Integration with Data Loading
- [ ] Add idempotency checks to data loading patterns
- [ ] Skip already processed files
- [ ] Log skipped files
- [ ] Update processed count in audit logs

**Acceptance Criteria:**
- Files are checked before processing
- Already processed files are skipped with log message
- File checksums are calculated and stored
- State file is updated after successful processing
- Failed processing does not mark file as processed

### Phase 4: Audit Logging (Week 4-5)

#### 4.1 AuditLogger Class
- [ ] Implement `AuditLogger` class
- [ ] Add session management with unique IDs
- [ ] Implement JSON-based audit logs
- [ ] Add progress tracking (discovered, processed, skipped)
- [ ] Add session start/end methods

**Acceptance Criteria:**
- Session ID format: `{SYMBOL}_{YYYYMMDD_HHMMSS}`
- Audit logs in `audit_logs/` directory
- Tracks: session metadata, file counts, timestamps, status
- JSON format for easy parsing
- Session updates are atomic
- Failed sessions are marked appropriately

#### 4.2 Audit Integration
- [ ] Start audit session before processing
- [ ] Update progress during processing
- [ ] End session with final status
- [ ] Handle error scenarios in audit logging

**Acceptance Criteria:**
- Every processing run has an audit session
- Progress is updated in real-time
- Error scenarios are captured in audit logs
- Session cleanup happens in finally blocks
- Audit logs are human-readable JSON

### Phase 5: Data Loading Core (Week 5-6)

#### 5.1 ParquetDataLoader Class
- [ ] Implement main `ParquetDataLoader` class
- [ ] Integrate all components (discovery, idempotency, audit)
- [ ] Add range-based data loading methods
- [ ] Implement error handling and recovery
- [ ] Add configuration validation

**Acceptance Criteria:**
- Initialization requires valid config path
- All methods support range parameters
- Error handling is comprehensive
- Integration with existing Chronos components
- Methods are idempotent by default

#### 5.2 Data Loading Methods
- [ ] Implement `load_training_data()` with ranges
- [ ] Implement `load_prediction_data()` with ranges
- [ ] Implement `load_incremental_data()` with ranges
- [ ] Add data quality validation
- [ ] Add feature engineering integration

**Acceptance Criteria:**
- All methods support year/month ranges
- Idempotency is enforced automatically
- Data quality checks are applied
- Error handling is consistent
- Integration with Chronos format conversion

### Phase 6: Integration & Testing (Week 6-7)

#### 6.1 Chronos Integration
- [ ] Update `ChronosTrainer` to use parquet loader
- [ ] Update `ChronosLoader` to use parquet loader
- [ ] Add configuration path parameters
- [ ] Maintain backward compatibility

**Acceptance Criteria:**
- Existing code continues to work
- New parquet functionality is optional
- Configuration is explicit (no defaults)
- Error handling is consistent

#### 6.2 Testing
- [ ] Unit tests for all components
- [ ] Integration tests with sample data
- [ ] Error scenario testing
- [ ] Performance testing with large datasets
- [ ] Idempotency testing

**Acceptance Criteria:**
- 90%+ code coverage
- All error scenarios are tested
- Idempotency is verified
- Performance meets requirements
- Integration tests pass

## Development Checklist

### Pre-Development Setup
- [ ] Create feature branch: `feature/parquet-loader`
- [ ] Set up development environment
- [ ] Create test data directory structure
- [ ] Set up CI/CD pipeline for testing

### Core Development
- [ ] Implement configuration system
- [ ] Implement file discovery with ranges
- [ ] Implement idempotency tracking
- [ ] Implement audit logging
- [ ] Implement main loader class
- [ ] Add comprehensive error handling

### Integration
- [ ] Integrate with existing Chronos components
- [ ] Update configuration templates
- [ ] Add usage examples
- [ ] Update documentation

### Testing & Validation
- [ ] Unit tests for all components
- [ ] Integration tests
- [ ] Performance tests
- [ ] Error scenario tests
- [ ] Idempotency verification

### Documentation
- [ ] Update README with parquet loader usage
- [ ] Create configuration guide
- [ ] Add troubleshooting guide
- [ ] Update API documentation

## Acceptance Criteria Summary

### Functional Requirements
1. **Range Specification**: Support year/month ranges in all discovery methods
2. **Idempotency**: No reprocessing of already processed files
3. **Audit Logging**: Complete trace of all processing activities
4. **Error Handling**: Comprehensive error handling with clear messages
5. **Configuration**: Fail-fast configuration with no defaults

### Non-Functional Requirements
1. **Performance**: Handle large datasets efficiently
2. **Memory**: Optimize memory usage with chunked loading
3. **Reliability**: Robust error handling and recovery
4. **Maintainability**: Clean, well-documented code
5. **Testability**: Comprehensive test coverage

### Integration Requirements
1. **Chronos Compatibility**: Seamless integration with existing components
2. **Backward Compatibility**: Existing functionality remains unchanged
3. **Configuration**: Explicit configuration required
4. **Error Handling**: Consistent error handling patterns

## Testing Strategy

### Unit Tests
- Configuration validation
- File discovery with various ranges
- Idempotency tracking
- Audit logging
- Error handling scenarios

### Integration Tests
- End-to-end data loading
- Chronos integration
- Error recovery scenarios
- Performance with large datasets

### Manual Testing
- Configuration edge cases
- File system edge cases
- Error scenario validation
- User experience testing

## Success Metrics

1. **Functionality**: All features work as specified
2. **Performance**: Meets performance requirements
3. **Reliability**: Handles errors gracefully
4. **Usability**: Easy to configure and use
5. **Maintainability**: Code is clean and well-documented

## Risk Mitigation

1. **Complexity**: Keep implementation simple and focused
2. **Performance**: Test with realistic data sizes
3. **Integration**: Maintain backward compatibility
4. **Configuration**: Ensure fail-fast behavior
5. **Testing**: Comprehensive test coverage

## Timeline

- **Week 1-2**: Core infrastructure and configuration
- **Week 3-4**: File discovery and idempotency
- **Week 5-6**: Audit logging and data loading
- **Week 7**: Integration, testing, and documentation

Total: 7 weeks for complete implementation
