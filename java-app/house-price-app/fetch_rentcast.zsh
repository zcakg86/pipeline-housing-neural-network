#!/bin/zsh

set -euo pipefail

SCRIPT_DIR=${0:A:h}
cd "$SCRIPT_DIR"

usage() {
  cat <<'EOF'
Usage: ./fetch_rentcast.zsh --confirm [--limit=500] [--max-calls=1] [--offset=0]

Safe defaults:
  --limit=500
  --max-calls=1
  --offset=0

No RentCast request is made unless --confirm is present.
EOF
}

confirmed=false
limit=500
max_calls=1
offset=0

for argument in "$@"; do
  case "$argument" in
    --confirm)
      confirmed=true
      ;;
    --limit=<->)
      limit=${argument#--limit=}
      ;;
    --max-calls=<->)
      max_calls=${argument#--max-calls=}
      ;;
    --offset=<->)
      offset=${argument#--offset=}
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      print -u2 "Unknown or invalid argument: $argument"
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$confirmed" != true ]]; then
  print -u2 "No API request made. Add --confirm only when you are ready to use the RentCast call."
  usage >&2
  exit 2
fi

if (( limit < 1 || limit > 500 )); then
  print -u2 "--limit must be between 1 and 500"
  exit 2
fi

if (( max_calls < 1 || max_calls > 5 )); then
  print -u2 "--max-calls must be between 1 and 5"
  exit 2
fi

# The Maven project targets Java 21. Prefer the Homebrew JDK explicitly so an
# older JAVA_HOME inherited from the terminal cannot make compilation fail.
if [[ -x /opt/homebrew/opt/openjdk@21/bin/java ]]; then
  export JAVA_HOME=/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home
else
  detected_java_home=$(/usr/libexec/java_home -v 21 2>/dev/null || true)
  if [[ -z "$detected_java_home" || ! -x "$detected_java_home/bin/java" ]]; then
    print -u2 "Java 21 is required. Install it with: brew install openjdk@21"
    print -u2 "No API request was made."
    exit 1
  fi
  export JAVA_HOME=$detected_java_home
fi
export PATH="$JAVA_HOME/bin:$PATH"

java_version=$("$JAVA_HOME/bin/java" -version 2>&1 | head -n 1)
if [[ "$java_version" != *'"21.'* ]]; then
  print -u2 "Expected Java 21 but found: $java_version"
  print -u2 "No API request was made."
  exit 1
fi

print "Starting RentCast fetch: limit=$limit, max-calls=$max_calls, offset=$offset"
print "Java: $java_version"
print "Press Ctrl-C to stop. Calls are reserved before they are sent."

exec ./mvnw compile exec:java \
  -Dexec.mainClass=com.houseprices.cli.RentcastFetcher \
  -Dexec.args="--confirm --limit=$limit --max-calls=$max_calls --offset=$offset"
