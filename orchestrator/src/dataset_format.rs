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

/// Converts a delimited dataset file (CSV/TSV) to a raw packed `f32` binary file.
///
/// Reads the source file line by line without loading it into memory,
/// parses each field as `f32`, and writes its 4-byte little-endian
/// representation directly to the output file.
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
/// any field cannot be parsed as `f32`.
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

    for (line_n, line) in reader.lines().enumerate() {
        let line = line?;
        let trimmed = line.trim();

        if trimmed.is_empty() {
            continue;
        }

        for field in trimmed.split(separator) {
            let val: f32 = field.trim().parse().map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "line {}: cannot parse field as f32: {:?}",
                        line_n + 1,
                        field
                    ),
                )
            })?;
            writer.write_all(&val.to_le_bytes())?;
        }
    }

    writer.flush()?;

    info!(
        "conversion complete: {} bytes written to {}",
        std::fs::metadata(&out)?.len(),
        out.display()
    );

    Ok(out)
}