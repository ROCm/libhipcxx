import os
import socket
import sys

# Ensure that this is being run on a specific platform
assert (
    sys.platform.startswith("linux")
    or sys.platform.startswith("darwin")
    or sys.platform.startswith("cygwin")
    or sys.platform.startswith("freebsd")
    or sys.platform.startswith("netbsd")
)


def env_path():
    ep = os.environ.get("LIBCXX_FILESYSTEM_DYNAMIC_TEST_ROOT")
    assert ep is not None
    ep = os.path.realpath(ep)
    assert os.path.isdir(ep)
    return ep


env_path_global = env_path()


# Make sure we don't try and write outside of env_path.
# All paths used should be sanitized
def sanitize(p):
    """Ensure path is within the test environment directory.

    Security fix: Use os.path.commonpath() instead of commonprefix()
    to properly validate path hierarchy, not just string prefixes.
    Raises ValueError (not assert) to work correctly even with -O flag.
    """
    p = os.path.realpath(p)
    env_path = os.path.realpath(env_path_global)

    # Check if p is within env_path using path comparison
    try:
        os.path.relpath(p, env_path)
    except ValueError:
        # On Windows, relpath raises ValueError if paths are on different drives
        raise ValueError(f"Path {p} is not within test root {env_path}")

    # Ensure the resolved path actually starts with the env_path
    # Use os.path.commonpath which works correctly with path components
    try:
        if os.path.commonpath([env_path, p]) != env_path:
            raise ValueError(f"Path {p} escapes test root {env_path}")
    except ValueError as e:
        # commonpath can raise ValueError for incompatible paths (different drives, etc.)
        raise ValueError(f"Path {p} is incompatible with test root {env_path}: {e}")

    return p


"""
Some of the tests restrict permissions to induce failures.
Before we delete the test environment, we have to walk it and re-raise the
permissions.
"""


def clean_recursive(root_p):
    if not os.path.islink(root_p):
        os.chmod(root_p, 0o777)  # nosec B103 - Test cleanup needs permissive permissions
    for ent in os.listdir(root_p):
        p = os.path.join(root_p, ent)
        if os.path.islink(p) or not os.path.isdir(p):
            os.remove(p)
        else:
            assert os.path.isdir(p)
            clean_recursive(p)
            os.rmdir(p)


def init_test_directory(root_p):
    root_p = sanitize(root_p)
    assert not os.path.exists(root_p)
    os.makedirs(root_p)


def destroy_test_directory(root_p):
    root_p = sanitize(root_p)
    clean_recursive(root_p)
    os.rmdir(root_p)


def create_file(fname, size):
    with open(sanitize(fname), "w") as f:
        f.write("c" * size)


def create_dir(dname):
    os.mkdir(sanitize(dname))


def create_symlink(source, link):
    os.symlink(sanitize(source), sanitize(link))


def create_hardlink(source, link):
    os.link(sanitize(source), sanitize(link))


def create_fifo(source):
    os.mkfifo(sanitize(source))


def create_socket(source):
    sock = socket.socket(socket.AF_UNIX)
    sanitized_source = sanitize(source)
    # AF_UNIX sockets may have very limited path length, so split it
    # into chdir call (with technically unlimited length) followed
    # by bind() relative to the directory
    os.chdir(os.path.dirname(sanitized_source))
    sock.bind(os.path.basename(sanitized_source))


# Security fix: Replace eval() with explicit dispatch table to prevent code injection
# Dispatch table mapping command names to functions
ALLOWED_COMMANDS = {
    'init_test_directory': init_test_directory,
    'destroy_test_directory': destroy_test_directory,
    'create_file': create_file,
    'create_dir': create_dir,
    'create_symlink': create_symlink,
    'create_hardlink': create_hardlink,
    'create_fifo': create_fifo,
    'create_socket': create_socket,
}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.stderr.write("Error: Command name required\n")
        sys.exit(1)

    cmd_name = sys.argv[1]
    args = sys.argv[2:]

    if cmd_name not in ALLOWED_COMMANDS:
        sys.stderr.write(f"Error: Unknown command '{cmd_name}'\n")
        sys.exit(1)

    try:
        func = ALLOWED_COMMANDS[cmd_name]
        # Convert arguments based on command signature
        # Only create_file takes a numeric size parameter as second argument
        converted_args = []
        for i, arg in enumerate(args):
            if cmd_name == 'create_file' and i == 1:
                # Second argument of create_file is the size (int)
                try:
                    converted_args.append(int(arg))
                except ValueError:
                    raise ValueError(f"create_file size must be numeric, got: {arg}")
            else:
                # All other arguments are paths (strings)
                converted_args.append(arg)

        func(*converted_args)
        sys.exit(0)
    except Exception as e:
        sys.stderr.write(f"Error executing {cmd_name}: {e}\n")
        sys.exit(1)
