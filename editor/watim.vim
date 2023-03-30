" Mostly stolen from https://gitlab.com/tsoding/porth/-/blob/master/editor/porth.vim
"
" Vim syntax file
" Language: Watim

" Usage Instructions
" Put this file in .vim/syntax/watim.vim
" and add in your .vimrc file the next line:
" autocmd BufRead,BufNewFile *.porth set filetype=porth

if exists("b:current_syntax")
  finish
endif

syntax keyword watimTodos TODO FIXME NOTE

" Language keywords
syntax keyword watimKeywords struct if extern loop local fn memory break else import as

" Comments
syntax region watimCommentLine start="//" end="$" contains=watimTodos

" String literals
syntax region watimString start=/\v"/ skip=/\v\\./ end=/\v"/ contains=watimEscapes

" Escape literals \n, \r, ....
syntax match watimEscapes display contained "\\[nr\"', tr\"']"

" Number literals
syn match watimNumber "\<\d\+\>"

syntax match watimVariable "$[0-9,a-z,A-Z,\-,*,_]*"
syntax match watimAssign "#[0-9,a-z,A-Z,\-,*,_]*"
syntax match watimRef "&[0-9,a-z,A-Z,\-,*,_]*"
syntax match watimStore ">>[0-9,a-z,A-Z,\-,*,_]*"
syntax match watimDeclare "@[0-9,a-z,A-Z,\-,*,_]*"

syntax keyword watimPrimitiveTypes i64 i32 bool

" Set highlights
highlight default link watimTodos Todo
highlight default link watimKeywords Keyword
highlight default link watimCommentLine Comment
highlight default link watimString String
highlight default link watimNumber Constant
highlight default link watimPrimitiveTypes Type
highlight default link watimChar Character
highlight default link watimEscapes SpecialChar
highlight default link watimVariable Identifier
highlight default link watimAssign Identifier
highlight default link watimRef Identifier
highlight default link watimStore Identifier
highlight default link watimDeclare Identifier
highlight default link watimWord Function

let b:current_syntax = "watim"
