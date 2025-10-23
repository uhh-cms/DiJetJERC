#!/bin/sh
action() {
    local shell_is_zsh="$( [ -z "${ZSH_VERSION}" ] && echo "false" || echo "true" )"
    local this_file="$( ${shell_is_zsh} && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local this_dir="$( cd "$( dirname "${this_file}" )" && pwd )"

    if ! ${DIJET_SETUP}; then
        source setup.sh dev
    fi

    target="${1:-${this_dir}/dijet}"
    shift

    flake8 --config "${this_dir}/.flake8" "${target}" "$@"
}

action "$@"
