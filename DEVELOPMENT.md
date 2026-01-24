# Naari Studio VTON - Development Guide

## Project Architecture

### Deployment
- **Primary**: HuggingFace Spaces (https://huggingface.co/spaces/GaneshGowri/naari-avatar)
- **GitHub**: This repository for version control and rollback tags

### Module Structure
```
naari-avatar/
├── app.py              # Gradio UI interface
├── config.py           # Centralized configuration
├── engine_idmvton.py   # IDM-VTON API integration
├── preprocess.py       # Image preprocessing utilities
├── requirements.txt    # Python dependencies
└── saree_draping/      # Future: Saree draping module
```

## Version History

### v3.0.0-garment-working (Current Anchor)
- IDM-VTON based garment transfer working
- 85% success rate across test photos
- HF_TOKEN authentication implemented

### Known Issues
1. **Ponytail Artifact**: Dark backgrounds misinterpreted as hair
2. **Finger Rendering**: Incomplete hand/finger reproduction
3. **Color Accuracy**: Garment colors occasionally differ

## Development Workflow

### Making Changes
1. Edit files directly on HuggingFace Spaces
2. Test the changes in the live Space
3. If successful, create a new release tag here
4. If issues arise, rollback to previous tag

### Rollback Procedure
```bash
# To rollback to working state:
# 1. Copy the code from the tagged release
# 2. Replace files on HuggingFace Space
# 3. Restart the Space
```

### Testing Checklist
- [ ] Male photo - front facing
- [ ] Female photo - front facing
- [ ] Various skin tones
- [ ] Different backgrounds
- [ ] Multiple garment types

## Next Steps

### Phase 1: Garment Improvements
- Fix ponytail/background artifact
- Improve finger rendering
- Color accuracy enhancement

### Phase 2: Jewelry Module
- Pose estimation integration
- Necklace, earrings, bangles support
- High precision for fine details

### Phase 3: Advanced Features
- Professional lighting effects
- Background customization
- 360° view from video

## Contact
Built by AnahataSri (Naari Studio) for e-commerce virtual try-on.
