terraform {
  backend "s3" {
    # Supply institution/account-specific settings with -backend-config.
    # Local validation uses init -backend=false and never contacts remote state.
  }
}
