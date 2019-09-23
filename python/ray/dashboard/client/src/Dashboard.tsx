import Link from "@material-ui/core/Link";
import { Theme } from "@material-ui/core/styles/createMuiTheme";
import createStyles from "@material-ui/core/styles/createStyles";
import withStyles, { WithStyles } from "@material-ui/core/styles/withStyles";
import Table from "@material-ui/core/Table";
import TableBody from "@material-ui/core/TableBody";
import TableCell from "@material-ui/core/TableCell";
import TableHead from "@material-ui/core/TableHead";
import TableRow from "@material-ui/core/TableRow";
import Typography from "@material-ui/core/Typography";
import AddIcon from "@material-ui/icons/Add";
import RemoveIcon from "@material-ui/icons/Remove";
import classNames from "classnames";
import React from "react";
import { Route } from "react-router";
import { Link as RouterLink } from "react-router-dom";
import Errors from "./Errors";
import Logs from "./Logs";
import UsageBar from "./UsageBar";

const formatByteAmount = (amount: number, unit: "mebibyte" | "gibibyte") =>
  `${(
    amount / (unit === "mebibyte" ? Math.pow(1024, 2) : Math.pow(1024, 3))
  ).toFixed(1)} ${unit === "mebibyte" ? "MiB" : "GiB"}`;

const formatUsage = (
  used: number,
  total: number,
  unit: "mebibyte" | "gibibyte"
) => {
  const usedFormatted = formatByteAmount(used, unit);
  const totalFormatted = formatByteAmount(total, unit);
  const percent = (100 * used) / total;
  return `${usedFormatted} / ${totalFormatted} (${percent.toFixed(0)}%)`;
};

const formatUptime = (bootTime: number) => {
  const uptimeSecondsTotal = Date.now() / 1000 - bootTime;
  const uptimeSeconds = Math.floor(uptimeSecondsTotal) % 60;
  const uptimeMinutes = Math.floor(uptimeSecondsTotal / 60) % 60;
  const uptimeHours = Math.floor(uptimeSecondsTotal / 60 / 60) % 24;
  const uptimeDays = Math.floor(uptimeSecondsTotal / 60 / 60 / 24);
  const pad = (value: number) => value.toString().padStart(2, "0");
  return [
    uptimeDays ? `${uptimeDays}d` : "",
    `${pad(uptimeHours)}h`,
    `${pad(uptimeMinutes)}m`,
    `${pad(uptimeSeconds)}s`
  ].join(" ");
};

const styles = (theme: Theme) =>
  createStyles({
    root: {
      backgroundColor: theme.palette.background.paper,
      padding: theme.spacing(2),
      "& > :not(:first-child)": {
        marginTop: theme.spacing(2)
      }
    },
    cell: {
      padding: theme.spacing(1),
      textAlign: "center",
      "&:last-child": {
        paddingRight: theme.spacing(1)
      }
    },
    expandCollapseCell: {
      cursor: "pointer"
    },
    expandCollapseIcon: {
      color: theme.palette.text.secondary,
      fontSize: "1.5em",
      verticalAlign: "middle"
    },
    cpuUsage: {
      minWidth: 60
    },
    secondary: {
      color: theme.palette.text.secondary
    }
  });

// TODO(mitchellstern): Add JSON schema validation for the node info.
interface NodeInfo {
  clients: Array<{
    now: number;
    hostname: string;
    ip: string;
    boot_time: number;
    cpu: number;
    cpus: [number, number];
    mem: [number, number, number];
    disk: {
      [path: string]: {
        total: number;
        free: number;
        used: number;
        percent: number;
      };
    };
    load_avg: [[number, number, number], [number, number, number]];
    net: [number, number];
    workers: Array<{
      pid: number;
      create_time: number;
      name: string;
      cmdline: string[];
      cpu_percent: number;
      cpu_times: {
        system: number;
        children_system: number;
        user: number;
        children_user: number;
      };
      memory_info: {
        pageins: number;
        pfaults: number;
        vms: number;
        rss: number;
      };
      memory_full_info: null;
    }>;
  }>;
  logs: {
    [ip: string]: {
      [pid: string]: string[];
    };
  };
  errors: {
    [jobId: string]: Array<{
      message: string;
      timestamp: number;
      type: string;
    }>;
  };
}

interface State {
  response: {
    result: NodeInfo;
    timestamp: number;
  } | null;
  error: string | null;
  expanded: {
    [hostname: string]: boolean;
  };
}

class Component extends React.Component<WithStyles<typeof styles>, State> {
  state: State = {
    response: null,
    error: null,
    expanded: {}
  };

  fetchNodeInfo = async () => {
    try {
      const url = new URL(
        "/api/node_info",
        process.env.NODE_ENV === "development"
          ? "http://localhost:8080"
          : window.location.href
      );
      const response = await fetch(url.toString());
      const json = await response.json();
      this.setState({ response: json, error: null });
    } catch (error) {
      this.setState({ response: null, error: error.toString() });
    } finally {
      setTimeout(this.fetchNodeInfo, 1000);
    }
  };

  toggleExpand = (hostname: string) => () => {
    this.setState(state => ({
      expanded: {
        ...state.expanded,
        [hostname]: !state.expanded[hostname]
      }
    }));
  };

  async componentDidMount() {
    await this.fetchNodeInfo();
  }

  render() {
    const { classes } = this.props;
    const { response, error, expanded } = this.state;

    if (error !== null) {
      return (
        <Typography className={classes.root} color="error">
          {error}
        </Typography>
      );
    }

    if (response === null) {
      return (
        <Typography className={classes.root} color="textSecondary">
          Loading...
        </Typography>
      );
    }

    const { result, timestamp } = response;

    const logCounts: {
      [hostname: string]: {
        perWorker: {
          [pid: string]: number;
        };
        total: number;
      };
    } = {};

    const errorCounts: {
      [hostname: string]: {
        perWorker: {
          [pid: string]: number;
        };
        total: number;
      };
    } = {};

    for (const client of result.clients) {
      logCounts[client.hostname] = { perWorker: {}, total: 0 };
      errorCounts[client.hostname] = { perWorker: {}, total: 0 };
      for (const worker of client.workers) {
        logCounts[client.hostname].perWorker[worker.pid] = 0;
        errorCounts[client.hostname].perWorker[worker.pid] = 0;
      }
    }

    for (const ip of Object.keys(result.logs)) {
      let hostname: string | null = null;
      for (const client of result.clients) {
        if (ip === client.ip) {
          hostname = client.hostname;
          break;
        }
      }
      if (hostname !== null) {
        for (const pid of Object.keys(result.logs[ip])) {
          const logCount = result.logs[ip][pid].length;
          if (pid in logCounts[hostname].perWorker) {
            logCounts[hostname].perWorker[pid] = logCount;
          }
          logCounts[hostname].total += logCount;
        }
      }
    }

    for (const jobErrors of Object.values(result.errors)) {
      for (const error of jobErrors) {
        const match = error.message.match(/\(pid=(\d+), host=(.*?)\)/);
        if (match !== null) {
          const pid = match[1];
          const hostname = match[2];
          if (hostname in errorCounts) {
            if (pid in errorCounts[hostname].perWorker) {
              errorCounts[hostname].perWorker[pid]++;
            }
            errorCounts[hostname].total++;
          }
        }
      }
    }

    const ipToHostname: { [ip: string]: string } = {};
    for (const client of result.clients) {
      ipToHostname[client.ip] = client.hostname;
    }

    return (
      <div className={classes.root}>
        <Typography variant="h5">Ray Dashboard</Typography>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell className={classes.cell} />
              <TableCell className={classes.cell}>Hostname</TableCell>
              <TableCell className={classes.cell}>Workers</TableCell>
              <TableCell className={classes.cell}>Uptime</TableCell>
              <TableCell className={classes.cell}>CPU</TableCell>
              <TableCell className={classes.cell}>RAM</TableCell>
              <TableCell className={classes.cell}>Disk</TableCell>
              {/*<TableCell className={classes.cell}>Sent</TableCell>*/}
              {/*<TableCell className={classes.cell}>Received</TableCell>*/}
              <TableCell className={classes.cell}>Logs</TableCell>
              <TableCell className={classes.cell}>Errors</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {result.clients.map(client => {
              return (
                <React.Fragment key={client.hostname}>
                  <TableRow hover>
                    <TableCell
                      className={classNames(
                        classes.cell,
                        classes.expandCollapseCell
                      )}
                      onClick={this.toggleExpand(client.hostname)}
                    >
                      {!expanded[client.hostname] ? (
                        <AddIcon className={classes.expandCollapseIcon} />
                      ) : (
                        <RemoveIcon className={classes.expandCollapseIcon} />
                      )}
                    </TableCell>
                    <TableCell className={classes.cell}>
                      {client.hostname}
                    </TableCell>
                    <TableCell className={classes.cell}>
                      {client.workers.length}
                    </TableCell>
                    <TableCell className={classes.cell}>
                      {formatUptime(client.boot_time)}
                    </TableCell>
                    <TableCell className={classes.cell}>
                      <div className={classes.cpuUsage}>
                        <UsageBar
                          percent={client.cpu}
                          text={`${client.cpu.toFixed(1)}%`}
                        />
                      </div>
                    </TableCell>
                    <TableCell className={classes.cell}>
                      <UsageBar
                        percent={
                          (100 * (client.mem[0] - client.mem[1])) /
                          client.mem[0]
                        }
                        text={formatUsage(
                          client.mem[0] - client.mem[1],
                          client.mem[0],
                          "gibibyte"
                        )}
                      />
                    </TableCell>
                    <TableCell className={classes.cell}>
                      <UsageBar
                        percent={
                          (100 * client.disk["/"].used) / client.disk["/"].total
                        }
                        text={formatUsage(
                          client.disk["/"].used,
                          client.disk["/"].total,
                          "gibibyte"
                        )}
                      />
                    </TableCell>
                    {/*<TableCell className={classes.cell}>{(client.net[0] / Math.pow(1024, 2)).toFixed(3)} MiB/s</TableCell>*/}
                    {/*<TableCell className={classes.cell}>{(client.net[1] / Math.pow(1024, 2)).toFixed(3)} MiB/s</TableCell>*/}
                    <TableCell className={classes.cell}>
                      {logCounts[client.hostname].total === 0 ? (
                        <span className={classes.secondary}>No logs</span>
                      ) : (
                        <Link
                          component={RouterLink}
                          to={`/logs/${client.hostname}`}
                        >
                          View all logs (
                          {logCounts[client.hostname].total.toLocaleString()}{" "}
                          {logCounts[client.hostname].total === 1
                            ? "line"
                            : "lines"}
                          )
                        </Link>
                      )}
                    </TableCell>
                    <TableCell className={classes.cell}>
                      {errorCounts[client.hostname].total === 0 ? (
                        <span className={classes.secondary}>No errors</span>
                      ) : (
                        <Link
                          component={RouterLink}
                          to={`/errors/${client.hostname}`}
                        >
                          View all errors (
                          {errorCounts[client.hostname].total.toLocaleString()})
                        </Link>
                      )}
                    </TableCell>
                  </TableRow>
                  {expanded[client.hostname] &&
                    client.workers.map((worker, index: number) => (
                      <TableRow hover key={index}>
                        <TableCell className={classes.cell} />
                        <TableCell className={classes.cell}>
                          {worker.cmdline[0].split(":", 2)[0]} (PID:{" "}
                          {worker.pid})
                        </TableCell>
                        <TableCell className={classes.cell}>
                          {worker.cmdline[0].split(":", 2)[1] || (
                            <span className={classes.secondary}>Idle</span>
                          )}
                        </TableCell>
                        <TableCell className={classes.cell}>
                          {formatUptime(worker.create_time)}
                        </TableCell>
                        <TableCell className={classes.cell}>
                          <UsageBar
                            percent={worker.cpu_percent}
                            text={`${worker.cpu_percent.toFixed(1)}%`}
                          />
                        </TableCell>
                        <TableCell className={classes.cell}>
                          <UsageBar
                            percent={
                              (100 * worker.memory_info.rss) / client.mem[0]
                            }
                            text={formatByteAmount(
                              worker.memory_info.rss,
                              "mebibyte"
                            )}
                          />
                        </TableCell>
                        <TableCell className={classes.cell}>
                          <span className={classes.secondary}>
                            Not available
                          </span>
                        </TableCell>
                        <TableCell className={classes.cell}>
                          {logCounts[client.hostname].perWorker[worker.pid] ===
                          0 ? (
                            <span className={classes.secondary}>No logs</span>
                          ) : (
                            <Link
                              component={RouterLink}
                              to={`/logs/${client.hostname}/${worker.pid}`}
                            >
                              View log (
                              {logCounts[client.hostname].perWorker[
                                worker.pid
                              ].toLocaleString()}{" "}
                              {logCounts[client.hostname].perWorker[
                                worker.pid
                              ] === 1
                                ? "line"
                                : "lines"}
                              )
                            </Link>
                          )}
                        </TableCell>
                        <TableCell className={classes.cell}>
                          {errorCounts[client.hostname].perWorker[
                            worker.pid
                          ] === 0 ? (
                            <span className={classes.secondary}>No errors</span>
                          ) : (
                            <Link
                              component={RouterLink}
                              to={`/errors/${client.hostname}/${worker.pid}`}
                            >
                              View errors (
                              {errorCounts[client.hostname].perWorker[
                                worker.pid
                              ].toLocaleString()}
                              )
                            </Link>
                          )}
                        </TableCell>
                      </TableRow>
                    ))}
                </React.Fragment>
              );
            })}
          </TableBody>
        </Table>
        <Typography align="center">
          Last updated: {new Date(timestamp * 1000).toLocaleString()}
        </Typography>
        <Route
          path="/logs/:hostname/:pid?"
          render={props => (
            <Logs {...props} ipToHostname={ipToHostname} logs={result.logs} />
          )}
        />
        <Route
          path="/errors/:hostname/:pid?"
          render={props => <Errors {...props} errors={result.errors} />}
        />
      </div>
    );
  }
}

export default withStyles(styles)(Component);
