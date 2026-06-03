//! Integration tests for the compiler self-upgrade functionality.

use iris::upgrade::SemVer;

#[test]
fn test_integration_semver_parsing() {
    let current = SemVer::parse("v0.6.0").expect("parse current");
    let remote = SemVer::parse("0.6.1").expect("parse remote");
    let remote_patch = SemVer::parse("v0.6.2-alpha").expect("parse alpha");

    assert!(remote > current);
    assert!(remote_patch > remote);
    assert_eq!(current.major, 0);
    assert_eq!(current.minor, 6);
    assert_eq!(current.patch, 0);
}

#[test]
fn test_integration_semver_invalid() {
    assert!(SemVer::parse("v").is_none());
    assert!(SemVer::parse("0").is_none());
    assert!(SemVer::parse("invalid-version").is_none());
}
