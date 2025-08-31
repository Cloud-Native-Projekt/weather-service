# Build Service

A Docker-based build service for creating Python wheel distributions of the weather service packages. This service provides a standardized, reproducible build environment for packaging the `weather_models` and `openmeteo_client` libraries.

## Overview

The `build_service` creates distributable Python wheels for the weather service core packages using a multi-stage Docker build process. It ensures consistent builds across different environments and provides a clean separation between build dependencies and final artifacts.

### Build Environment

- **Python 3.13.7**: Latest stable Python runtime on Debian Trixie
- **Poetry 2.1**: Modern dependency management and packaging

### Build Process Flow

1. **Environment Setup**: Install Poetry and build tools
2. **Source Copy**: Copy package source directories
3. **Wheel Building**: Build distributable wheels for each package
4. **Artifact Collection**: Consolidate wheels in output directory
5. **Final Stage**: Extract wheels to scratch image

## Built Packages

### weather_models

- **Purpose**: Core data models, database management, and ML capabilities
- **Version**: As specified in `pyproject.toml`
- **Output**: `weather_models-*.whl`

### openmeteo_client

- **Purpose**: OpenMeteo API client with rate limiting and data processing
- **Version**: As specified in `pyproject.toml`
- **Output**: `openmeteo-client-*.whl`

## Usage

### Building Wheels

```bash
# Build the service and extract wheels
docker build --target weather-build-service -t weather-build-service .

# Create container to extract wheels
docker create --name wheel-extractor weather-build-service

# Copy wheels to host
docker cp wheel-extractor:/wheels ./dist/

# Cleanup
docker rm wheel-extractor
```

### CI/CD Integration

```yaml
# Example GitHub Actions workflow
- name: Build Python wheels
  run: |
    docker build --target weather-build-service -t wheels .
    docker create --name extractor wheels
    docker cp extractor:/wheels ./dist/
    docker rm extractor

- name: Upload artifacts
  uses: actions/upload-artifact@v3
  with:
    name: python-wheels
    path: dist/*.whl
```

## Output Artifacts

### Wheel Files

The build service produces standard Python wheel files:

- **Format**: `{package}-{version}-py3-none-any.whl`
- **Location**: `/wheels/` in final image
- **Compatibility**: Python 3.13+ on any platform

### Example Output

```
wheels/
├── weather_models-x.y.z-py3-none-any.whl
└── openmeteo_client-x.y.z-py3-none-any.whl
```

### Package Updates

When updating package versions:

1. Update version in respective `pyproject.toml`
2. Rebuild using Docker build service
3. Test new wheels in target environment
4. Update deployment configurations with new versions