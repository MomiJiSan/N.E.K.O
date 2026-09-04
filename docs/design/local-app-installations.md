# Local App installation registry

N.E.K.O keeps executable locations separate from plugin manifests. A plugin may
declare `[plugin.local_app]` authorization and operation mappings, but it cannot
choose a program path or process arguments.

Before starting N.E.K.O, set `NEKO_LOCAL_APP_INSTALLATIONS_FILE` to the absolute
path of a host-owned JSON file. The file is read once before the loopback bridge
listener starts; later plugin or renderer input cannot change it.

```json
{
  "version": 1,
  "installations": [
    {
      "app_id": "knowledge_dungeon",
      "title": "Knowledge Dungeon",
      "executable": "C:\\Program Files\\N.E.K.O Apps\\Knowledge Dungeon\\Knowledge Dungeon.exe",
      "args": []
    }
  ]
}
```

The executable must be an existing absolute regular-file path. Arguments are
fixed by this file. An installation is enabled only when an enabled plugin also
declares the same `app_id`; unknown IDs fail closed. The browser UI sends only
the `app_id`. Pairing material is delivered to the child solely through stdin.

## Independent Electron repository in development

For a development checkout, the trusted file may point at the locally installed
Node executable and Electron CLI, keeping the independent repository path in
fixed arguments.
For example, replace these generic paths with absolute paths on the host:

```json
{
  "version": 1,
  "installations": [
    {
      "app_id": "knowledge_dungeon",
      "title": "Knowledge Dungeon",
      "executable": "C:\\Program Files\\nodejs\\node.exe",
      "args": [
        "D:\\src\\N.E.K.O-Knowledge-Dungeon\\node_modules\\electron\\cli.js",
        "D:\\src\\N.E.K.O-Knowledge-Dungeon"
      ]
    }
  ]
}
```

Then point N.E.K.O at the file before startup:

```powershell
$env:NEKO_LOCAL_APP_INSTALLATIONS_FILE = 'D:\neko-config\local-app-installations.json'
```

The Electron main process must read exactly one JSON line from stdin for its
launch material. It must not expect the launch code in arguments or environment
variables. If the independent project uses another Electron entry, change only
the host-owned fixed `args` array.
