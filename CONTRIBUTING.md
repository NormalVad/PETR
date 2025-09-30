# Contributing to PETRv2 Multi-Frame Adaptive Extensions

Thank you for your interest in contributing to this project! This document provides guidelines for contributing to the PETRv2 Multi-Frame Adaptive Extensions repository.

## 🚀 Getting Started

### Prerequisites
- Python 3.6.8+
- PyTorch 1.9.0+
- CUDA 11.2+
- mmdetection3d 0.17.1+
- Basic understanding of 3D object detection and temporal fusion

### Development Setup
1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/PETRv2-Adaptive-Extensions.git`
3. Install dependencies: `pip install -r requirements.txt`
4. Setup mmdetection3d: `cd mmdetection3d && pip install -v -e . && cd ..`

## 📝 Types of Contributions

### Bug Reports
- Use the issue template
- Provide detailed reproduction steps
- Include system information and error logs
- Attach relevant code snippets

### Feature Requests
- Describe the feature clearly
- Explain the use case and benefits
- Consider backward compatibility
- Provide implementation suggestions if possible

### Code Contributions
- Follow the existing code style
- Add appropriate documentation
- Include tests for new functionality
- Update README.md if needed

## 🔧 Development Guidelines

### Code Style
- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings for all functions and classes
- Keep functions focused and modular

### Testing
- Test your changes thoroughly
- Add unit tests for new functionality
- Ensure existing tests still pass
- Test on different configurations (3-frame, 4-frame)

### Documentation
- Update docstrings for modified functions
- Add comments for complex logic
- Update README.md for new features
- Include usage examples

## 🏗️ Project Structure

```
PETRv2-Adaptive-Extensions/
├── projects/
│   ├── configs/           # Configuration files
│   └── mmdet3d_plugin/    # Custom implementations
├── tools/
│   └── misc/              # Visualization and analysis tools
├── slurm_train_*.sh       # Training scripts
├── visualize_*.py         # Visualization scripts
└── results.py            # Analysis tools
```

## 🧪 Testing Your Changes

### Local Testing
```bash
# Test 3-frame adaptive model
bash slurm_train_3frame.sh

# Test 4-frame adaptive model
bash slurm_train_4frame.sh

# Test visualization tools
python tools/misc/visualize_adaptive_weights.py \
    projects/configs/petrv2/petrv2_3frame_adaptive.py \
    work_dirs/petrv2_3frame_adaptive/latest.pth \
    --show-dir test_vis --frames 3 --num-samples 2
```

### Validation
- Ensure training converges properly
- Verify visualization tools work correctly
- Check that adaptive weights are reasonable
- Validate performance metrics

## 📋 Pull Request Process

### Before Submitting
1. Ensure your code follows the style guidelines
2. Add appropriate tests
3. Update documentation
4. Test your changes thoroughly

### Pull Request Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Local testing completed
- [ ] Visualization tools tested
- [ ] Performance metrics validated

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
```

### Review Process
1. Maintainers will review your PR
2. Address any feedback promptly
3. Make necessary changes
4. Ensure CI passes (if applicable)

## 🐛 Reporting Issues

### Bug Reports
Use the bug report template and include:
- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- System information
- Relevant logs/error messages

### Feature Requests
Use the feature request template and include:
- Clear description of the feature
- Use case and motivation
- Proposed implementation (if any)
- Potential impact

## 📚 Resources

### Documentation
- [mmdetection3d Documentation](https://mmdetection3d.readthedocs.io/)
- [PETRv2 Paper](https://arxiv.org/abs/2206.01256)
- [Original PETR Repository](https://github.com/megvii-research/PETR)

### Getting Help
- Check existing issues and discussions
- Create a new issue for questions
- Join community discussions (if available)

## 🎯 Areas for Contribution

### High Priority
- Performance optimizations
- Additional visualization tools
- More adaptive weighting strategies
- Better documentation

### Medium Priority
- Support for more datasets
- Additional model architectures
- Enhanced analysis tools
- Tutorial notebooks

### Low Priority
- Code refactoring
- Style improvements
- Minor bug fixes
- Documentation updates

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

## 🙏 Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

Thank you for contributing to PETRv2 Multi-Frame Adaptive Extensions!
