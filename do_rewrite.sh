#!/usr/bin/env bash
export FILTER_BRANCH_SQUELCH_WARNING=1
cd "/c/Users/Moon/Desktop/Projects/IRIS"
git filter-branch --env-filter '
if [ "$GIT_AUTHOR_NAME" = "Moon9t" ]; then
  export GIT_AUTHOR_EMAIL="luyanduthandokhumalo@gmail.com"
  export GIT_COMMITTER_EMAIL="luyanduthandokhumalo@gmail.com"
fi
' --tag-name-filter cat -- --all
