use std::{
    fs::File,
    io::{self, BufRead, BufReader, BufWriter, Write},
    path::{Path, PathBuf},
};

use log::info;

/// Supported delimited dataset formats that can be transparently converted
/// to raw packed `f32` binary before training begins.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DelimitedFormat {
    /// Comma-separated values — no header, all columns are `f32`.
    Csv,
    /// Tab-separated values — no header, all columns are `f32`.
    Tsv,
}

impl DelimitedFormat {
    /// Detects the delimited format from the file extension.
    ///
    /// # Returns
    /// `Some(DelimitedFormat)` if the extension is a known delimited format,
    /// or `None` if the file should be treated as raw binary.
    pub fn from_path(path: &Path) -> Option<Self> {
        match path.extension().and_then(|e| e.to_str()) {
            Some("csv") => Some(Self::Csv),
            Some("tsv") => Some(Self::Tsv),
            _ => None,
        }
    }

    /// Returns the field separator byte for this format.
    fn separator(self) -> u8 {
        match self {
            Self::Csv => b',',
            Self::Tsv => b'\t',
        }
    }
}

/// Returns `true` if every non-empty field in `line` is parseable as `f32`.
///
/// Used to detect whether the first row of a delimited file is a header.
fn is_data_row(line: &str, separator: char) -> bool {
    line.split(separator)
        .all(|field| field.trim().parse::<f32>().is_ok())
}

/// Converts a delimited dataset file (CSV/TSV) to a raw packed `f32` binary file.
///
/// Reads the source file line by line without loading it into memory,
/// parses each field as `f32`, and writes its 4-byte little-endian
/// representation directly to the output file.
///
/// If the first non-empty row cannot be fully parsed as `f32` values it is
/// treated as a header and skipped automatically. No explicit configuration
/// is required.
///
/// The output file is placed next to the source file with a `.bin` extension,
/// replacing the original extension. If the output file already exists,
/// conversion is skipped and the existing path is returned immediately.
///
/// # Args
/// * `src` - Path to the source delimited file.
/// * `format` - The delimited format to use for parsing.
///
/// # Returns
/// The path to the converted binary file.
///
/// # Errors
/// Returns an `io::Error` if the file cannot be read or written, or if
/// any field in a data row cannot be parsed as `f32`.
pub fn convert_to_binary(src: &Path, format: DelimitedFormat) -> io::Result<PathBuf> {
    let out = src.with_extension("bin");

    if out.exists() {
        info!(
            "binary cache already exists at {}, skipping conversion",
            out.display()
        );
        return Ok(out);
    }

    info!(
        "converting {} to binary at {}",
        src.display(),
        out.display()
    );

    let separator = format.separator() as char;
    let reader = BufReader::new(File::open(src)?);
    let mut writer = BufWriter::new(File::create(&out)?);

    let mut lines = reader.lines().enumerate();

    // Detect and skip header: find the first non-empty line and check if it
    // parses as data. If not, log and skip it; if yes, process it normally.
    let first_data_line = loop {
        match lines.next() {
            None => break None,
            Some((_, Err(e))) => return Err(e),
            Some((n, Ok(line))) => {
                let trimmed = line.trim().to_owned();
                if trimmed.is_empty() {
                    continue;
                }
                if !is_data_row(&trimmed, separator) {
                    info!("skipping header row: {trimmed:?}");
                    break None;
                }
                break Some((n, trimmed));
            }
        }
    };

    // Process the first data line if it wasn't a header.
    if let Some((line_n, line)) = first_data_line {
        write_row(&line, separator, line_n + 1, &mut writer)?;
    }

    // Process remaining lines.
    for (line_n, line) in lines {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        write_row(trimmed, separator, line_n + 1, &mut writer)?;
    }

    writer.flush()?;

    info!(
        "conversion complete: {} bytes written to {}",
        std::fs::metadata(&out)?.len(),
        out.display()
    );

    Ok(out)
}

/// Parses a delimited row and writes each field as a little-endian `f32` to `writer`.
///
/// # Args
/// * `line` - The trimmed row string.
/// * `separator` - The field separator character.
/// * `line_n` - The 1-based line number, used in error messages.
/// * `writer` - The output writer.
///
/// # Errors
/// Returns an `io::Error` if any field cannot be parsed as `f32`.
fn write_row(line: &str, separator: char, line_n: usize, writer: &mut BufWriter<File>) -> io::Result<()> {
    for field in line.split(separator) {
        let val: f32 = field.trim().parse().map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "line {line_n}: cannot parse field as f32: {:?}",
                    field
                ),
            )
        })?;
        writer.write_all(&val.to_le_bytes())?;
    }
    Ok(())
}