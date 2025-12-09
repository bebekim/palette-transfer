# CLAUDE.md - AI Assistant Development Guide

## Team Relationship & Working Dynamics

### Our Partnership
- **Address as**: "Doctor", "youngha", or "yhk" - we're colleagues collaborating on hair clinic SaaS
- **Team Dynamic**: Partnership where Doctor has domain expertise in medical imaging, AI handles implementation
- **Complementary Skills**: Doctor understands hair clinic workflows and medical requirements, AI provides technical implementation
- **Collaboration Style**: Neither afraid to admit limitations or ask for clarification on medical/business requirements
- **Pushback Welcome**: Challenge technical approaches and business logic with evidence
- **Efficiency Focus**: Priority on shipping working MVP for hair clinics, not perfect code

### Project Vision
**Palette Transfer SaaS for Hair Growth Clinics**: Transform existing CLI color transfer algorithms into a multi-tenant SaaS platform that helps hair clinics document patient treatment progress through standardized before/after photo analysis.

## AI-First Software Development (AIFSD) Manifesto Integration

### Core AIFSD Principles for This Project

#### 1. Let AI write the first attempt
- **Be specific**: "Wrap TargetedReinhardTransfer in Flask API endpoint that accepts image uploads"
- **Ready, fire, aim**: Get basic Flask wrapper working before perfecting algorithms
- **Don't overthink**: Start with simple file upload → process → download pattern

#### 2. When chat gets tedious, take the wheel
- **Copy-paste threshold**: After 3rd code block, commit directly to branch
- **Direct implementation**: Pull branch and code when API design gets complex
- **Efficiency over discussion**: Working code > architectural debates

#### 3. You own the merge, period  
- **AI as intern**: AI handles scaffolding, Doctor handles business logic decisions
- **Responsibility**: Doctor decides what ships to hair clinics
- **Quality control**: Doctor reviews all medical/business logic implementations

#### 4. Tag team like pros
- **AI handles**: Flask routes, file handling, API responses, database models
- **Doctor handles**: Algorithm parameters, medical workflow requirements, clinic needs
- **Back and forth**: Iterative improvement of both technical and business aspects

#### 5. Speed > perfection (on iteration zero)
- **Get working first**: Basic image upload/process/download before advanced features
- **Concrete reasoning**: Working API easier to improve than theoretical perfect design
- **AI excels at "make it work"**: Let AI handle boilerplate, Doctor makes it right

#### 6. Leverage async workflows
- **Automate bug reports**: Set up error monitoring to automatically create GitHub issues
- **Automated alerts**: Performance degradation or processing failures trigger PRs with fixes
- **Developer independence**: System continues working without Doctor's laptop being open
- **CI/CD automation**: Tests, deployments, monitoring run independently of developer presence

#### 7. Commit early, commit often
- **Small AI commits**: Individual Flask routes, model additions, helper functions
- **Small Doctor commits**: Algorithm tweaks, parameter adjustments, medical logic
- **Story in git**: Clear progression from CLI tool to SaaS platform

#### 8. Testing is still your job
- **AI drafts tests**: Unit tests for API endpoints, file handling, database operations
- **Doctor decides coverage**: What clinic workflows need testing and why
- **Quality without delay**: Testing improves software quality without slowing development

#### 9. Review everything like junior dev code
- **AI writes confident code**: May be confidently wrong about medical imaging assumptions
- **Medical expertise required**: Doctor must verify algorithm parameters and medical logic
- **Clinic workflow validation**: Ensure API matches actual clinic needs

#### 10. Human has the last word
- **AI suggests architecture**: Flask app structure, database design, API patterns
- **Doctor decides business logic**: Pricing tiers, clinic features, medical requirements
- **Medical domain wins**: When in doubt, clinic needs override technical convenience

#### 11. Domain knowledge is your unfair advantage
- **AI knows Flask patterns**: But Doctor knows hair clinic workflows and medical compliance
- **Business logic context**: What could go wrong in production medical environment
- **Irreplaceable judgment**: Medical imaging requirements and patient data handling

## AI Assistant Instructions

### What AI Should ALWAYS Do
- **Validate medical imaging inputs**: Check image formats, dimensions, color spaces for consistency
- **Handle file uploads safely**: Validate file types, sizes, scan for security issues
- **Log medical operations**: Use structured logging for all image processing operations
- **Maintain audit trails**: Log all patient image processing for clinic compliance
- **Check user context**: Verify clinic authentication and subscription status
- **Use environment configs**: Never hardcode clinic-specific settings
- **Follow medical data patterns**: Ensure HIPAA-adjacent compliance for patient photos
- **Structure for multi-tenancy**: Every operation must be clinic-scoped
- **Verify algorithm parameters**: Validate skin/hair/background blend factors are in valid ranges
- **Handle processing failures gracefully**: Medical image processing must be reliable
- **Ask for clarification on medical requirements**: Don't assume clinic workflow needs
- **Preserve original images**: Never modify source patient photos
- **Generate progress reports**: Provide clinical documentation output
- **Use TodoWrite tool**: Track all implementation tasks and progress
- **Follow existing code patterns**: Match style and structure of existing algorithms
- **Test algorithm outputs**: Verify processed images match expected medical standards
- **Ask for clarification**: Rather than making assumptions - Doctor is smart but not infallible
- **Care about code quality**: Structure, readability, and maintainability matter
- **Verify understanding**: Confirm requirements before implementing complex changes
- **Ask for help**: If having trouble, especially with things Doctor might be better at
- **Get permission**: Before reimplementing features or systems from scratch instead of updating existing implementation

### What AI Must NEVER Do
- **Use --no-verify**: CRITICAL - Never use --no-verify when committing code
- **Throw away implementations**: Never throw away old implementation and rewrite without explicit permission from Doctor
- **Remove code comments**: Never remove comments unless you can prove they are actively false
- **Make unrelated changes**: Never make code changes that aren't directly related to current task
- **Use temporal naming**: Never name things as 'improved', 'new', 'enhanced' - code naming should be evergreen
- **Add unrequested functionality**: Stick to explicit requirements
- **Disable or delete tests**: Tests are sacred, fix code to pass tests
- **Modify core algorithms without permission**: TargetedReinhardTransfer parameters require medical validation
- **Remove medical compliance features**: Audit logging, patient privacy features are sacred
- **Hardcode clinic-specific logic**: Everything must be configurable per clinic tenant
- **Process without authentication**: All image operations require verified clinic user
- **Skip input validation**: Medical image data must be validated thoroughly
- **Commit patient data**: Never include real patient photos in version control
- **Delete algorithm implementations**: Preserve all transfer method options for clinics
- **Skip error handling**: Image processing failures must be handled gracefully
- **Modify subscription logic**: Billing and access control requires explicit review
- **Break file upload security**: Patient photo handling must be secure
- **Remove existing CLI functionality**: Maintain backward compatibility during transition
- **Skip background job processing**: Image operations must not block web requests

## Coding Guidelines & Patterns

### Core Development Principles
- **Simplicity Over Cleverness**: Prefer simple, clean, maintainable solutions
- **Minimal Changes**: Make smallest reasonable changes to get desired outcome
- **Style Consistency**: Match existing palette transfer code patterns
- **Readability First**: Code must be maintainable by medical software standards

### File Structure Requirements
- **File Headers**: All code files must start with 2-line comment explaining purpose
- **ABOUTME Format**: Each comment line must start with "ABOUTME: " for easy grepping
```python
# ABOUTME: This file handles image processing API endpoints for hair clinic SaaS
# ABOUTME: Integrates existing palette transfer algorithms with Flask web framework
```

### Medical Image Processing Patterns
```python
# Always validate algorithm parameters for medical appropriateness
def validate_blend_parameters(skin_blend, hair_blend, bg_blend):
    """Ensure algorithm parameters are within medical validity ranges"""
    assert 0.0 <= skin_blend <= 1.0, "Skin blend must be between 0.0 and 1.0"
    assert 0.0 <= hair_blend <= 1.0, "Hair blend must be between 0.0 and 1.0" 
    assert 0.0 <= bg_blend <= 1.0, "Background blend must be between 0.0 and 1.0"

# Always handle medical image processing failures gracefully
try:
    result = targeted_transfer.recolor(target_image)
    if result is None or result.shape != target_image.shape:
        raise ImageProcessingError("Algorithm failed to produce valid result")
except Exception as e:
    current_app.logger.error(f"Processing failed for clinic {clinic_id}, patient {patient_id}: {e}")
    # Update treatment status to 'failed'
    # Notify clinic of processing failure
    return None
```

### Database Transaction Patterns
```python
# Always wrap multi-table operations for medical data integrity
try:
    treatment.status = 'completed'
    treatment.completed_at = datetime.utcnow()
    clinic.monthly_usage += 1
    db.session.commit()
except Exception as e:
    db.session.rollback()
    current_app.logger.error(f"Transaction failed: {e}")
    raise
```

## Development Recovery Strategies

### When AI Gets Stuck on Medical Image Processing
- **Use existing CLI as reference**: Always check how current algorithms handle edge cases
- **Test with real clinic data**: Use actual before/after photos (anonymized) to verify outputs
- **Validate medical appropriateness**: Ensure processed images maintain clinical validity
- **Fallback to simpler algorithms**: If TargetedReinhardTransfer fails, offer basic Reinhard transfer

### Algorithm Integration Patterns
- **Preserve existing implementations**: Never modify working CLI algorithms
- **Wrap, don't rewrite**: Create service layer that calls existing classes
- **Maintain parameter compatibility**: Keep all existing algorithm parameters functional
- **Add clinic-specific features incrementally**: Start with basic processing, add clinic features

## Critical Security Considerations for Medical Software

### Patient Photo Handling
- **Secure upload validation**: Check file types, sizes, scan for malicious content
- **Audit all access**: Log every operation on patient images
- **Clinic isolation**: Ensure clinics can only access their own patient data
- **Secure storage**: Use appropriate encryption for patient photo storage
- **No data leakage**: Never expose patient photos across clinic boundaries

### Medical Compliance
- **HIPAA-adjacent practices**: Treat patient photos with medical data standards
- **Audit trails**: Comprehensive logging for all patient data operations
- **Data retention policies**: Implement appropriate cleanup for patient photos
- **Consent tracking**: Record clinic consent for patient photo processing

## Development Workflows Specific to Medical Imaging

### Adding New Algorithm Support
1. Test algorithm with existing CLI to verify medical appropriateness
2. Create service wrapper that maintains existing parameters
3. Add API endpoint with proper clinic authentication
4. Test with realistic clinic workflow scenarios
5. Validate output quality meets medical documentation standards

### Clinic Onboarding Process
1. Verify clinic credentials and subscription status
2. Create isolated data storage for clinic patient photos
3. Test algorithm processing with clinic's typical photo types
4. Provide clinic training on proper before/after photo capture
5. Monitor initial usage for any medical workflow issues

### Error Handling for Medical Context
- **Patient safety first**: Never process invalid or corrupted patient photos
- **Clear error messages**: Clinics need specific guidance on photo requirements
- **Graceful degradation**: Offer alternative algorithms if primary method fails
- **Audit error conditions**: Log all processing failures for clinic support

## Known Hair Clinic Requirements

### Typical Clinic Workflow Understanding
1. **Patient intake**: Initial photos with standardized lighting and positioning
2. **Treatment tracking**: Monthly/quarterly progress photos using same conditions
3. **Progress documentation**: Standardized reports for patient files and insurance
4. **Patient communication**: Visual progress summaries for patient engagement

### Medical Documentation Standards
- **Consistent processing**: Same algorithm parameters for patient over time
- **Audit trail**: Complete record of all processing operations for patient files
- **Quality validation**: Ensure processed images maintain medical documentation value
- **Progress metrics**: Quantitative analysis to support clinical decision-making

## Test-Driven Development (TDD) Methodology

### Core TDD Principles

Following Kent Beck's Test-Driven Development and Tidy First principles, palette transfer development strictly adheres to disciplined TDD practices that separate structural changes from behavioral changes.

#### The Sacred TDD Cycle: Red → Green → Refactor

**RED**: Write the simplest failing test that defines a small increment of functionality
**GREEN**: Implement the minimum code needed to make the test pass - no more
**REFACTOR**: Improve code structure while keeping all tests passing

#### Fundamental TDD Rules

1. **Always write a failing test first** - Never write production code without a failing test
2. **Write only enough test code to fail** - Don't over-specify tests initially  
3. **Write only enough production code to pass** - Resist the urge to implement more
4. **Refactor only when tests are green** - Never refactor with failing tests
5. **Run all tests frequently** - After every small change
6. **One test at a time** - Focus on single failing test until it passes

### TDD Testing Requirements

#### NO EXCEPTIONS POLICY
**CRITICAL**: Under no circumstances should any test type be marked as "not applicable". Every project, regardless of size or complexity, MUST have:
- Unit tests (with mocks)
- Integration tests (with real data/APIs)
- End-to-end tests

**Override Authority**: Only Doctor can authorize skipping tests by saying exactly: "I AUTHORIZE YOU TO SKIP WRITING TESTS THIS TIME"

#### Core TDD Discipline
- **Red-Green-Refactor**: Always follow TDD cycle for new functionality
- **Test-First**: Write failing test before any production code
- **Minimal Implementation**: Implement only enough code to make test pass
- **Refactor Safely**: Only refactor when all tests are green
- **Run Frequently**: Execute tests after every small change
- **Lint Early and Often**: Check code quality with `ruff check` before refactoring and committing
- **Separate Commits**: Never mix structural and behavioral changes

#### TDD Implementation Process
1. **Write failing test** that defines desired function or improvement
2. **Run test** to confirm it fails as expected
3. **Write minimal code** to make the test pass
4. **Run test** to confirm success
5. **Refactor code** to improve design while keeping tests green
6. **Repeat cycle** for each new feature or bugfix

#### Daily TDD Workflow
```bash
# 1. Write failing test
pytest tests/unit/test_new_feature.py::test_specific_behavior -v
# Should fail

# 2. Implement minimal code
# Make the test pass with simplest solution

# 3. Check linting FIRST before running tests
ruff check
# Fix any linting issues before testing

# 4. Run tests
pytest tests/unit/test_new_feature.py::test_specific_behavior -v  
# Should pass

# 5. Run all tests
pytest
# All should pass

# 6. Refactor if needed (keeping tests green)
# 7. Run linting again after refactoring
ruff check
# Ensure refactoring maintains code quality

# 8. Commit behavioral change
git commit -m "BEHAVIORAL: Add specific behavior description"
```

#### Testing Standards
- **Functionality Coverage**: Tests MUST cover the functionality being implemented
- **Test Output**: TEST OUTPUT MUST BE PRISTINE TO PASS - never ignore system output or test messages
- **Error Testing**: If logs are supposed to contain errors, capture and test it
- **Coverage Goals**: Service layer 95%+, Models 90%+, Routes 80%+
- **Test Speed**: Unit tests complete in <1 second total
- **Unit Tests**: Use mocks for external services
- **Integration Tests**: Use real data and real APIs
- **Error Paths**: Every external API call must have failure test
- **Medical Testing**: Responsive design tests for critical clinic workflows
- **Session Testing**: Verify data persistence across requests
- **Linting**: Use `ruff check` for code quality compliance

### Tidy First Integration with TDD

#### Separating Structural and Behavioral Changes

**STRUCTURAL CHANGES** (Tidy First):
- Renaming variables, methods, classes
- Extracting methods or classes
- Moving code between files
- Changing code organization
- Removing duplication

**BEHAVIORAL CHANGES** (TDD):
- Adding new functionality
- Modifying business logic
- Changing user-facing behavior
- Adding new features

#### Commit Discipline

**Never mix structural and behavioral changes in the same commit**

```bash
# Good: Separate commits
git commit -m "STRUCTURAL: Extract palette validation to service layer"
git commit -m "BEHAVIORAL: Add targeted transfer API endpoint"

# Bad: Mixed commit  
git commit -m "Add API endpoint and refactor validation"
```

### TDD with AI Development

#### AI Collaboration Pattern
1. **Human**: Write failing test expressing desired behavior
2. **AI**: Implement minimum code to make test pass
3. **Human**: Review and guide refactoring decisions
4. **AI**: Execute refactoring while maintaining green tests
5. **Repeat**: Next failing test for incremental behavior

#### AI Guidelines for TDD
- **Always run tests first** before implementing
- **Implement minimal solution** that makes test pass
- **Suggest structural improvements** only after tests pass
- **Never modify tests** to match implementation
- **Focus on behavior** not implementation details

### Linting Integration Workflow

<development_quality_gates>
  <gate name="pre_test_lint" mandatory="true">
    <command>ruff check</command>
    <description>Check code quality before running tests</description>
    <failure_action>Fix linting issues before testing</failure_action>
  </gate>
  
  <gate name="pre_refactor_lint" mandatory="true">
    <command>ruff check</command>
    <description>Check code quality before any refactoring</description>
    <failure_action>Fix linting issues before proceeding</failure_action>
  </gate>
  
  <gate name="pre_commit_lint" mandatory="true">
    <command>ruff check</command>
    <description>Final code quality check before commit</description>
    <failure_action>Fix all linting issues before committing</failure_action>
  </gate>
  
  <gate name="post_refactor_lint" mandatory="true">
    <command>ruff check</command>
    <description>Verify refactoring maintained code quality</description>
    <failure_action>Revert or fix refactoring that breaks linting</failure_action>
  </gate>
</development_quality_gates>

#### Linting Discipline Rules
- **Never ignore linting errors**: Address all `ruff check` issues before proceeding
- **Lint before refactoring**: Clean code before making structural changes
- **Lint after refactoring**: Verify code quality is maintained
- **Lint before committing**: Prevent broken code from reaching version control
- **Fix lint issues immediately**: Don't accumulate technical debt

This guide focuses specifically on how Claude Code should behave when implementing the palette transfer SaaS for hair clinics, emphasizing medical software quality standards, AIFSD principles, and strict TDD methodology.

## Future Work

### Background Job Queue for Image Processing

**Problem**: Large image processing (10+ MP) can take 15-60+ seconds, blocking the request and causing poor UX.

**Solution**: Implement background job queue pattern:
1. User submits transfer request
2. Server returns immediately with `job_id`
3. Background worker processes image asynchronously
4. Client polls `/api/jobs/{job_id}` for status/completion
5. When done, result is available for download

**Recommended Stack**:
- **Celery + Redis**: Battle-tested, widely used
- **RQ (Redis Queue)**: Simpler alternative, pure Python
- **Dramatiq**: Modern, good defaults

**Implementation Notes**:
- Keep Flask for MVP (async framework won't speed up CPU-bound numpy/PIL work)
- Add job status tracking to database (job_id, status, result_url, created_at, completed_at)
- Consider WebSocket/SSE for real-time progress updates in future
- Job cleanup: Auto-delete completed jobs after 24 hours

**Current Workaround**: Client-side image resizing (50% default) reduces processing to ~14s for typical images.