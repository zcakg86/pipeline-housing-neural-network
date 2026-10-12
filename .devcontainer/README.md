# Codespaces development environment

This configuration creates a Python 3.11 and Java 21 development environment
for model work and the Quarkus application. It is suitable for running and
testing the app through a private Codespaces forwarded port; it is not a
production hosting configuration.

## Start

1. Commit the `.devcontainer` directory and every required Java model artifact.
2. Configure `HF_TOKEN` under GitHub **Settings → Codespaces → Secrets** when
   the agent should use Hugging Face Inference Providers. Configure
   `CARTO_API_KEY` for the application basemap. Optionally configure
   `RENTCAST_API_KEY` and `ZILLOW_API_KEY` for their explicit ingestion actions.
   The application does not call these services merely because a Codespace
   starts, and secret values should never be committed to the repository.
3. Create the Codespace and wait for `post-create.sh` to finish.
4. Run `make java-dev` in the Codespace terminal.
5. Open the privately forwarded **Quarkus house-price app** port 8080.

The setup creates `.venv`, installs the pinned Python requirements and editable
project package, and warms the Maven dependency cache. It does not train models,
refresh market data, fetch property APIs, or deploy artifacts automatically.
